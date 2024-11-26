import numpy as np
from cvxopt import matrix, solvers
import concurrent.futures


class DataSet():
    """A class holding the training and testing data for training and evaluation; often load as the Model's data"""

    def __init__(self, train_by_user, train_by_item, test, num_batches):
        """Creates a DataSet and batches it.
        Args:
          train_by_user: list of (user, items)
          train_by_item: list of (item, users)
          test: list of (user, history_items, target_items)
          num_batches: partitions each set using this many batches.
        """
        self.train_by_user = train_by_user
        self.train_by_item = train_by_item
        self.test = test
        self.num_users = len(train_by_user)
        self.num_items = len(train_by_item)
        self.user_batches = self._batch(train_by_user, num_batches)
        self.item_batches = self._batch(train_by_item, num_batches)
        self.test_batches = self._batch(test, num_batches)

    def _batch(self, data, num_batches):
        batches = [[] for _ in range(num_batches)]  # TODO I would like to shuffle the data and split for just one time
        for i, records in enumerate(data):
            batches[i % num_batches].append(records)  # like put the data in a bin in sequence each time
        return batches


def map_parallel(fn, xs, *args):
    """Applies a function to a list, equivalent to [fn(x, *args) for x in xs].
    In IALS:
    fn: A function that can be solve a batch of user(item)'s profiles
    xs: User records divided into several groups for parallel training
    """
    if len(xs) == 1:
        return [fn(xs[0], *args)]  # if only one batch, it means no parallel learning

    num_threads = len(xs)
    executor = concurrent.futures.ProcessPoolExecutor(num_threads)  # initialize a process pool
    futures = [executor.submit(fn, x, *args) for x in xs]  # submit sub-tasks
    concurrent.futures.wait(futures)  # waiting for all processes down
    results = [future.result() for future in futures]  # retreive result of each sub-task
    return results


class IALS():
    def __init__(self, user_nums, item_nums, embedding_dim, reg, unobserved_weight, stddev, epsilon):
        """
        :param user_nums:
        :param item_nums:
        :param embedding_dim:
        :param reg:
        :param unobserved_weight:
        :param stddev:
        note: num is not maximum index!
        """
        self.embedding_dim = embedding_dim
        self.reg = reg
        self.unobserved_weight = unobserved_weight
        self.epsilon = epsilon
        self.user_embedding = np.random.normal(0, stddev, (user_nums, embedding_dim))  # TODO what is stddev
        self.item_embedding = np.random.normal(0, stddev, (item_nums, embedding_dim))
        self._update_user_gramian()
        self._update_item_gramian()
        self.sens = np.zeros(user_nums)

    def _update_user_gramian(self):
        self.user_gramian = np.matmul(self.user_embedding.T, self.user_embedding)

    def _update_item_gramian(self):
        self.item_gramian = np.matmul(self.item_embedding.T, self.item_embedding)

    def _update_sensitivity(self):
        vec_1_norm = np.sum(np.abs(self.user_embedding), axis=1)  # hold the 1_norm for all pu
        mat_1_norm = np.array([np.sum(np.abs(np.outer(self.user_embedding[i], self.user_embedding[i]))) for i in
                               range(self.user_embedding.shape[0])])  # loop through all user embedding

        # DPOCCF1
        # s1 = np.max(2 * vec_1_norm + mat_1_norm + 1)
        # s2 = np.max(2 * vec_1_norm + mat_1_norm)
        # s3 = np.max(2 * vec_1_norm)
        # print(s1, s2, s3)
        # self.sens = 3

        # DPOCCF
        self.sens = np.max(2 * vec_1_norm + mat_1_norm + 1)

    def train(self, ds):  # train for one epoch
        """Train one iteration for user and item embedding
        Args:
            ds: A dataset object
        """
        # update user_embedding
        self._solve(ds.user_batches, is_user=True)
        self._update_user_gramian()

        # update item_embedding
        self._solve(ds.item_batches, is_user=False)
        self._update_item_gramian()

    def _solve(self, batches, is_user=True):
        """Update the user or item embedding"""
        if is_user:  # solve the user's embedding
            embedding = self.user_embedding
            args = (self.item_embedding, self.item_gramian, self.reg, self.unobserved_weight)
        else:
            embedding = self.item_embedding
            args = (self.user_embedding, self.user_gramian, self.reg, self.unobserved_weight)
        results = map_parallel(solve, batches, *args)  # parallel training
        for r in results:
            for user, emb in r.items():  # the result r is a dict
                embedding[user, :] = emb  # update the embeddings involved in the batch

    def train_private(self, ds, epsilon):
        # update item_embedding
        self._update_sensitivity()
        self._solve_private(ds.item_batches, epsilon)
        # self._update_item_gramian()

    def _solve_private(self, batches, epsilon):
        embedding = self.item_embedding
        args = (self.user_embedding, self.user_gramian, self.reg, self.unobserved_weight, self.sens, epsilon)
        results = map_parallel(solve_private, batches, *args)  # parallel training
        for r in results:
            for user, emb in r.items():  # the result r is a dict
                embedding[user, :] = emb  # update the embeddings involved in the batch


def project(user_history, item_embedding, item_gramian, reg, unobserved_weight):
    """Solve the embedding for users or items"""
    if not user_history:
        raise ValueError("empty user_history in projection")

    emb_dim = item_embedding.shape[1]
    item_embedding_pos = item_embedding[user_history]  # see the formula in the manuscript

    lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + \
          unobserved_weight * item_gramian + \
          reg * np.identity(emb_dim)
    rhs = np.matmul(item_embedding_pos.T,
                    np.ones(shape=(item_embedding_pos.shape[0],)))  # Noted the shape should be 1d
    return np.linalg.solve(lhs, rhs)


def solve(data_by_user, item_embedding, item_gramian, global_reg, unobserved_weight):
    """This fun receive a batch of records and update the corresponding embedding, let's assume the user side"""
    user_embedding_updated = {}  # recording the users' embedding will be updated
    for uid, items in data_by_user:  # this can be data_by_item as well
        reg = global_reg * (len(items) + unobserved_weight * item_embedding.shape[0])
        user_embedding_updated[uid] = project(items, item_embedding, item_gramian, reg, unobserved_weight)
    return user_embedding_updated


def project_private(user_history, item_embedding, item_gramian, reg, unobserved_weight, sens, epsilon):
    """Solve the embedding for users or items"""
    if not user_history:
        raise ValueError("empty user_history in projection")

    emb_dim = item_embedding.shape[1]
    item_embedding_pos = item_embedding[user_history]  # see the formula in the manuscript
    # print(sens)

    # create noise vec
    noise_vec = np.random.laplace(0, sens / epsilon, size=emb_dim)

    # create symmetric noise mat
    noise_mat_full = np.random.laplace(0, sens / epsilon, size=(emb_dim, emb_dim))
    noise_mat = np.triu(noise_mat_full) + np.tril(noise_mat_full.T, -1)

    # create full noise mat
    # noise_mat_full = np.random.laplace(0, sens / epsilon, size=(emb_dim, emb_dim))
    # noise_mat = noise_mat_full

    # compute the noisy term in objective function
    # real_lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + \
    #            unobserved_weight * item_gramian + \
    #            reg * np.identity(emb_dim)
    # real_rhs = np.matmul(item_embedding_pos.T,
    #                      np.ones(shape=(item_embedding_pos.shape[0],)))  # Noted the shape should be 1d

    lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + noise_mat + \
          unobserved_weight * item_gramian + \
          reg * np.identity(emb_dim)
    rhs = np.matmul(item_embedding_pos.T, np.ones(shape=(item_embedding_pos.shape[0],))) + (1 / 2) * noise_vec.reshape(
        -1, )

    # inspect the relative magnitude between mat and noisy mat
    mat_rm = np.linalg.norm(lhs - (lhs - noise_mat))  # default is f-norm for mat
    vec_rm = np.linalg.norm(rhs - (rhs - (1 / 2) * noise_vec))  # default is 2-norm for vec
    return qp_solver(lhs, -rhs), mat_rm, vec_rm  # solve based on spectral trimming


def solve_private(data_by_user, item_embedding, item_gramian, global_reg, unobserved_weight, sens, epsilon):
    """This fun receive a batch of records and update the corresponding embedding, let's assume the user side"""
    user_embedding_updated = {}  # recording the users' embedding will be updated
    # inspect
    mat_rm_arr = []
    vec_rm_arr = []
    for uid, items in data_by_user:  # this can be data_by_item as well
        # compute and perturbed the reg
        reg = global_reg * (
                    len(items) + unobserved_weight * item_embedding.shape[0] + np.random.laplace(0, sens / epsilon))
        user_embedding_updated[uid], mat_rm, vec_rm = project_private(items, item_embedding, item_gramian, reg,
                                                                      unobserved_weight,
                                                                      sens, epsilon)
        mat_rm_arr.append(mat_rm)
        vec_rm_arr.append(vec_rm)
    # print(f"mat_rm:{np.average(mat_rm_arr)}")
    # print(f"vec_rm:{np.average(vec_rm_arr)}")
    return user_embedding_updated


def spectral_trimming(P):
    v, WT = np.linalg.eig(P)
    # print(v)
    # print(WT)
    W = WT.T
    W_ = W[v >= 0, :]
    v_ = v[v >= 0]
    v_diag = np.diag(v_)
    return v_diag, W_


# def qp_solver(C2, C1):
#     P = matrix(C2, tc='d')  # d表示double精度
#     q = matrix(C1, tc='d')
#     try:  # TODO change to if else control
#         assert np.all(np.linalg.eigvals(P) >= 0), 'P must be semidefinite'
#     except AssertionError as reason:
#         v_trimmed, W_trimmed = spectral_trimming(C2)
#         C2_trimmed = W_trimmed.T @ v_trimmed @ W_trimmed
#         C1_trimmed = C1 @ W_trimmed.T @ W_trimmed
#         # C2_trimmed = v_trimmed
#         # C1_trimmed = C1 @ W_trimmed.T
#         P = matrix(C2_trimmed, tc='d')  # d表示double精度
#         q = matrix(C1_trimmed, tc='d')
#         print(reason.__class__.__name__, reason)
#     result = solvers.qp(P, q)  # TODO why add minus?
#     # x = np.linalg.solve(P, -q).flatten()
#     x = np.array(result['x']).flatten()
#     return x


def qp_solver(C2, C1):
    P = matrix(C2, tc='d')  # d表示double精度
    q = matrix(C1, tc='d')
    if np.all(np.linalg.eigvals(P) >= 0):
        result = solvers.qp(P, q)
        x = np.array(result['x']).flatten()
    else:
        v_trimmed, W_trimmed = spectral_trimming(C2)
        C2_trimmed = v_trimmed
        C1_trimmed = C1 @ W_trimmed.T
        P = matrix(C2_trimmed, tc='d')  # d表示double精度
        q = matrix(C1_trimmed, tc='d')
        result = solvers.qp(P, q)
        temp = np.array(result['x']).flatten()
        x = np.linalg.lstsq(W_trimmed, temp, rcond=None)[0]  # solve the Ax=b
    return x

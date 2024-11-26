import numpy as np
# from cvxopt import matrix, solvers
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
import pandas as pd


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


def laplace_function(x, epsilon, sens):
    beta = sens / epsilon
    return np.exp(-np.abs(x) / beta) / (2 * beta)


def check_noise_distribution(s1, s2, s3, epsilon=0.1):
    print(s1, s2, s3)
    import seaborn as sns
    sns.set()
    x = np.linspace(-10, 10, 10000)
    y1 = np.vectorize(laplace_function)(x, epsilon, s1)
    y2 = np.vectorize(laplace_function)(x, epsilon, s2)
    y3 = np.vectorize(laplace_function)(x, epsilon, s3)
    # plt.plot(x, y1, linestyle='-', color=mcolors.TABLEAU_COLORS[colors[0]], label="DPOCCF1")
    # plt.plot(x, y2, linestyle='-', color=mcolors.TABLEAU_COLORS[colors[1]], label="DPOCCF2")
    # plt.plot(x, y3, linestyle='-', color=mcolors.TABLEAU_COLORS[colors[2]], label="DPOCCF3")
    # plt.plot(x, y1, linestyle='-', color=mcolors.TABLEAU_COLORS[colors[0]], label="DPIMF"+r'${\rm _{\\text{str}}}$')
    plt.plot(x, y1, linestyle='dotted', linewidth=3, color="seagreen", label=r'$\mathrm{DPIMF}_{str}$')
    plt.plot(x, y2, linestyle='dashed', linewidth=3, color="mediumslateblue", label=r"$\mathrm{DPIMF}_{com}$")
    plt.plot(x, y3, linestyle='solid', linewidth=3, color="royalblue", label=r"$\mathrm{DPIMF}_{sym}$")
    # plt.xlim(-20, 20, 10)
    plt.xlim(-10, 10, 10)
    # plt.ylim(0.001, 0.018, 10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Noise', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.legend(loc='upper right', prop={'size': 13})
    # plt.grid()
    plt.tight_layout()
    plt.show()


class IALS():
    def __init__(self, user_nums, item_nums, embedding_dim, reg, unobserved_weight, stddev, epsilon, sens_type='4',
                 clipping=False):
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
        self.sens_type = sens_type
        self.sens = []
        self.clipping = clipping

    def _update_sensitivity(self):
        """compute the sensitivity for the coefficients of the objective function"""
        vec_1_norm = np.sum(np.abs(self.user_embedding), axis=1)  # hold the 1_norm for all pu
        mat_1_norm_full = np.array([np.sum(np.abs(np.outer(self.user_embedding[i], self.user_embedding[i]))) for i in
                                    range(self.user_embedding.shape[0])])  # compute the 1_norm of the outer results
        mat_1_norm_triu = np.array(
            [np.sum(np.abs(np.triu(np.outer(self.user_embedding[i], self.user_embedding[i])))) for i in
             range(self.user_embedding.shape[0])])  # compute the 1_norm of the outer results

        # compute the noise distribution
        sens_2ord_1 = mat_1_norm_full
        sens_2ord_2 = (1 - 0.8) * mat_1_norm_full
        sens_2ord_3 = (1 - 0.8) * mat_1_norm_triu
        sens_2ord_1, sens_2ord_2, sens_2ord_3 = max(sens_2ord_1), max(sens_2ord_2), max(sens_2ord_3)
        check_noise_distribution(sens_2ord_1, sens_2ord_2, sens_2ord_3)
        exit()

        # compute the sensitivity
        # sens = 2 * vec_1_norm + (1 - self.unobserved_weight) * mat_1_norm_triu
        # sens = np.max(sens)  # compute the maximum sensitivity over all users
        # self.sens = sens

        # compute the sensitivity for DPOCCF2
        if self.sens_type == '2':
            sens_1ord = np.max(2 * vec_1_norm)
            sens_2ord = np.max((1 - self.unobserved_weight) * mat_1_norm_full)
            self.sens = [sens_1ord, sens_2ord, 1 - self.unobserved_weight]

        if self.sens_type == '3':
            sens_1ord = np.max(2 * vec_1_norm)
            sens_2ord = np.max((1 - self.unobserved_weight) * mat_1_norm_triu)
            self.sens = [sens_1ord, sens_2ord, 1 - self.unobserved_weight]

        if (self.sens_type == '4') or (self.unobserved_weight >= 1):
            sens_1ord = np.max(2 * vec_1_norm)
            sens_2ord = 0
            self.sens = [sens_1ord, sens_2ord, 0]

        print(self.sens)
        # exit()

    def train(self, ds):  # train for one epoch
        """Train one iteration for user and item embedding
        Args:
            ds: A dataset object
        """
        # update user_embedding
        self._solve(ds.user_batches, is_user=True)

        # update item_embedding
        self._solve(ds.item_batches, is_user=False)

    def _solve(self, batches, is_user=True):
        """Update the user or item embedding"""
        if is_user:  # solve the user's embedding
            embedding = self.user_embedding
            args = (self.item_embedding, self.reg, self.unobserved_weight)
        else:
            embedding = self.item_embedding
            args = (self.user_embedding, self.reg, self.unobserved_weight)
        results = map_parallel(solve, batches, *args)  # parallel training
        for r in results:
            for user, emb in r.items():  # the result r is a dict
                embedding[user, :] = emb  # update the embeddings involved in the batch

    def train_private(self, ds, epsilon):
        # update item_embedding
        self._update_sensitivity()
        self._solve_private(ds.item_batches, epsilon)

    def _solve_private(self, batches, epsilon):
        embedding = self.item_embedding
        args = (self.user_embedding, self.reg, self.unobserved_weight, self.sens, epsilon, self.clipping)
        results = map_parallel(solve_private, batches, *args)  # parallel training
        for r in results:
            for user, emb in r.items():  # the result r is a dict
                embedding[user, :] = emb  # update the embeddings involved in the batch


def project(user_history, item_embedding, reg, unobserved_weight):
    """Solve the embedding for users or items"""
    if not user_history:
        raise ValueError("empty user_history in projection")

    emb_dim = item_embedding.shape[1]
    user_history_unobserved = np.setdiff1d(np.arange(item_embedding.shape[0]),
                                           user_history)  # get the missing entries indexes
    item_embedding_pos = item_embedding[user_history]  # see the formula in the manuscript
    item_embedding_unobserved = item_embedding[user_history_unobserved]  # TODO serve like item_gramian

    lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + \
          unobserved_weight * np.matmul(item_embedding_unobserved.T, item_embedding_unobserved) + \
          reg * np.identity(emb_dim)
    rhs = np.matmul(item_embedding_pos.T,
                    np.ones(shape=(item_embedding_pos.shape[0],)))  # Noted the shape should be 1d
    return np.linalg.solve(lhs, rhs)


def solve(data_by_user, item_embedding, global_reg, unobserved_weight):
    """This fun receive a batch of records and update the corresponding embedding, let's assume the user side"""
    user_embedding_updated = {}  # recording the users' embedding will be updated
    for uid, items in data_by_user:  # this can be data_by_item as well
        reg = global_reg * (len(items) + unobserved_weight * (item_embedding.shape[0] - len(items)))
        user_embedding_updated[uid] = project(items, item_embedding, reg, unobserved_weight)
    return user_embedding_updated


def project_private(user_history, item_embedding, reg, unobserved_weight, sens, epsilon):
    """Solve the embedding for users or items"""
    if not user_history:
        raise ValueError("empty user_history in projection")
    emb_dim = item_embedding.shape[1]
    user_history_unobserved = np.setdiff1d(np.arange(item_embedding.shape[0]),
                                           user_history)  # get the missing entries indexes
    item_embedding_pos = item_embedding[user_history]  # see the formula in the manuscript
    item_embedding_unobserved = item_embedding[user_history_unobserved]  # TODO serve like item_gramian

    # compute the epsilon of each part
    if unobserved_weight < 1:
        epsilon_v = epsilon * 0.8
        epsilon_m = epsilon * 0.1
    else:
        epsilon_v = epsilon
        epsilon_m = epsilon

    sens_v, sens_m = sens[:-1]

    # create noise vec
    noise_vec = np.random.laplace(0, sens_v / epsilon_v, size=emb_dim)

    # create symmetric noise mat
    noise_mat_full = np.random.laplace(0, sens_m / epsilon_m, size=(emb_dim, emb_dim))
    noise_mat = np.triu(noise_mat_full) + np.tril(noise_mat_full.T, -1)

    # create full noise mat
    # noise_mat = noise_mat_full

    # compute the noisy term in objective function
    # real_lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + \
    #            unobserved_weight * item_gramian + \
    #            reg * np.identity(emb_dim)
    # real_rhs = np.matmul(item_embedding_pos.T,
    #                      np.ones(shape=(item_embedding_pos.shape[0],)))  # Noted the shape should be 1d
    if unobserved_weight >= 1:
        lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + unobserved_weight * np.matmul(
            item_embedding_unobserved.T, item_embedding_unobserved) + reg * np.identity(emb_dim)
    else:
        lhs = np.matmul(item_embedding_pos.T, item_embedding_pos) + unobserved_weight * np.matmul(
            item_embedding_unobserved.T, item_embedding_unobserved) + noise_mat + reg * np.identity(emb_dim)
    # rhs = np.matmul(item_embedding_pos.T + noise_vec.reshape(-1, 1),
    #                 np.ones(shape=(item_embedding_pos.shape[0],)))  # Noted the shape should be 1d
    # temp = np.matmul(item_embedding_pos.T, np.ones(shape=(item_embedding_pos.shape[0],)))
    rhs = np.matmul(item_embedding_pos.T, np.ones(shape=(item_embedding_pos.shape[0],))) + (1 / 2) * noise_vec.reshape(
        -1, )  # Noted the shape should be 1d
    return qp_solver(lhs, -rhs)  # solve based on spectral trimming


def solve_private(data_by_user, item_embedding, global_reg, unobserved_weight, sens, epsilon, clipping=False):
    """This fun receive a batch of records and update the corresponding embedding, let's assume the user side"""
    user_embedding_updated = {}  # recording the users' embedding will be updated
    L2norm_bd = np.sqrt(1 / global_reg)
    out_bd_nums = 0
    for uid, items in data_by_user:  # this can be data_by_item as well

        # compute the reg
        if unobserved_weight < 1:
            epsilon_3 = epsilon * 0.1
            eta = np.random.laplace(sens[-1] / epsilon_3)
            reg = global_reg * (len(items) + unobserved_weight * (item_embedding.shape[0] - len(items)) + eta)
        else:
            reg = global_reg * (len(items) + unobserved_weight * (item_embedding.shape[0] - len(items)))

        # solve the embedding
        embedding = project_private(items, item_embedding, reg,
                                    unobserved_weight,
                                    sens, epsilon)

        # clipping L2norm
        if clipping and np.linalg.norm(embedding) > L2norm_bd:
            out_bd_nums += 1
            embedding = embedding * (np.linalg.norm(embedding) / L2norm_bd)

        # update the embeddings
        user_embedding_updated[uid] = embedding
    print(f"out_bd_nums:{out_bd_nums}")
    return user_embedding_updated


def spectral_trimming(P):
    v, WT = np.linalg.eig(P)
    W = WT.T
    W_ = W[v >= 0, :]
    v_ = v[v >= 0]
    v_diag = np.diag(v_)
    return v_diag, W_


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

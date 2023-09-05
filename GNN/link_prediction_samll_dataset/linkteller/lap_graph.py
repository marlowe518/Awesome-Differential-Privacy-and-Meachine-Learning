import time

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity / eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2 * np.log(1.25 / delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise


def perturb_adj_continuous(adj, target_epsilon, target_delta, noise_type, noise_seed, n_split=50):
    """
    Args:
        self:
        adj: csc matrix of an undirected graph(symmetric matrix)

    Returns:
    """
    # convert to csr matrix
    adj = adj.tocsr()
    n_nodes = adj.shape[0]
    n_edges = len(adj.data) // 2

    N = n_nodes
    t = time.time()

    A = sp.tril(adj, k=-1)
    print('getting the lower triangle of adj matrix done!')

    eps_1 = target_epsilon * 0.01
    eps_2 = target_epsilon - eps_1
    noise = get_noise(noise_type=noise_type, size=(N, N), seed=noise_seed,
                      eps=eps_2, delta=target_delta, sensitivity=1)
    noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
    print(f'generating noise done using {time.time() - t} secs!')

    A += noise
    print(f'adding noise to the adj matrix done!')

    t = time.time()
    n_edges_keep = n_edges + int(
        get_noise(noise_type=noise_type, size=1, seed=noise_seed,
                  eps=eps_1, delta=target_delta, sensitivity=1)[0])
    print(f'edge number from {n_edges} to {n_edges_keep}')

    t = time.time()
    a_r = A.A.ravel()

    n_splits = n_splits
    len_h = len(a_r) // n_splits
    ind_list = []
    for i in tqdm(range(n_splits - 1)):
        ind = np.argpartition(a_r[len_h * i:len_h * (i + 1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * i) # 这里split之后，A矩阵中的元素被分割为多个len_h长的小batch，每个batch中找前topk个。这是为了避免全量找topk个的计算量。先减小数据规模。

    ind = np.argpartition(a_r[len_h * (n_splits - 1):], -n_edges_keep)[-n_edges_keep:]
    ind_list.append(ind + len_h * (n_splits - 1))

    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:] # 最后对取出来的数据找topk个

    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert (col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)
    print(f'data preparation done using {time.time() - t} secs!')

    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
    # Convert to csc matrix
    result = mat + mat.T
    result = result.tocsc()
    return result

import scipy.linalg
from torch_geometric.data import Data
import torch
import scipy.sparse as ssp
import numpy as np


def reverse_edges(edges):
    """Reverses an edgelist to obtain incoming edges for each node."""
    reversed_edges = {u: [] for u in edges}
    for u, u_neighbors in edges.items():
        for v in u_neighbors:
            reversed_edges[v].append(u)
    return reversed_edges


def get_adjacency_lists(dataset: Data):
    """Returns a dictionary of adjacency lists, with nodes as keys."""

    edges = {u: [] for u in range(dataset.num_nodes)}
    for u, v in dataset.edge_index.T.tolist():
        edges[u].append(v)
    return edges


def get_adjacency_lists_from_tensor(graph: torch.Tensor):
    ids = graph.unique().tolist()
    edges = {u: [] for u in ids}
    for u, v in graph.T.tolist():
        edges[u].append(v)
    return edges


def sample_adjacency_lists_undirected_graph(edges, train_nodes,
                                            max_degree, uniform=False):
    # edges: undirected edges
    # 无向图的特点是，如果a是b的入边，那么b也是a的入边，删除其中一个入边，另一个入边也需要删除
    # 这个函数与之前的区别是，这个函数在删除入边的时候，会删除对应的出边（反过来是对应若干节点的入边）
    # 该函数会动态地修改reverse_edges，比如，当节点1选择从入边中删除2，3后，2，3的入边集合也会将节点1删除
    # 这样可以保证2，3节点在筛选边的时候，避免选到会被后续删除的边。更重要的是直接保证了edges仍然是无向图的表示。
    train_nodes = set(train_nodes)
    all_nodes = edges.keys()

    reversed_edges = reverse_edges(edges)  # 逆邻接表记录每个节点的入边节点
    sampled_reversed_edges = {u: [] for u in all_nodes}
    dropped_count = 0
    dropped_users = []
    for u in all_nodes:
        incoming_edges = reversed_edges[u]  # 得到u所有邻居的id
        incoming_train_edges = [v for v in incoming_edges if v in train_nodes]  # 筛出包含在训练集中的节点
        if not incoming_train_edges:  # 只处理入边中训练集中的节点
            continue
        in_degree = len(incoming_train_edges)
        if not uniform:
            sampling_prob = max_degree / (2 * in_degree)
            # sampling_mask = (
            #         jax.random.uniform(u_rng, shape=(in_degree,)) <= sampling_prob)
            sampling_mask = (
                    np.random.uniform(size=(in_degree,)) <= sampling_prob)
            sampling_mask = np.asarray(sampling_mask)
        else:  # 取出max degree个边
            #
            if False:
                incoming_train_edges = np.asarray(incoming_train_edges)
                incoming_train_edges_lower = incoming_train_edges[np.where(incoming_train_edges > u)]
                incoming_train_edges_upper = incoming_train_edges[np.where(incoming_train_edges < u)]
                in_degree = len(incoming_train_edges_lower)
                sampling_mask = np.array([False] * in_degree)
                local_max_degree = max_degree - len(incoming_train_edges_upper)
                if local_max_degree < 0:
                    print("error")
                sample_size = min(in_degree, local_max_degree)
                inds = np.random.choice(np.arange(len(sampling_mask)), size=sample_size, replace=False)
                sampling_mask[inds] = True
                inverse_sampling_mask = ~sampling_mask  # 对sampling mask取反

                removed_incoming_train_edges = np.asarray(incoming_train_edges_lower)[inverse_sampling_mask]
                unique_removed_incoming_train_edges = np.unique(removed_incoming_train_edges)
                incoming_train_edges_lower = np.asarray(incoming_train_edges_lower)[sampling_mask]
                unique_incoming_train_edges = np.unique(incoming_train_edges_lower)
            else:
                incoming_train_edges = np.asarray(incoming_train_edges)
                sampling_mask = np.array([False] * in_degree)
                sample_size = min(in_degree, max_degree)
                inds = np.random.choice(np.arange(len(sampling_mask)), size=sample_size, replace=False)
                sampling_mask[inds] = True
                inverse_sampling_mask = ~sampling_mask  # 对sampling mask取反

                removed_incoming_train_edges = np.asarray(incoming_train_edges)[inverse_sampling_mask]
                unique_removed_incoming_train_edges = np.unique(removed_incoming_train_edges)
                incoming_train_edges = np.asarray(incoming_train_edges)[sampling_mask]
                unique_incoming_train_edges = np.unique(incoming_train_edges)
        # 对reversed_edges进行更新，将节点u sample out的边的对称边从reversed_edges中剔除
        for removed_edge in unique_removed_incoming_train_edges:
            reversed_edges[removed_edge].remove(u)
            if removed_edge < u:
                sampled_reversed_edges[removed_edge].remove(u)

        # Check that in-degree is bounded, otherwise drop this node.
        if len(unique_incoming_train_edges) <= max_degree:
            sampled_reversed_edges[u] = unique_incoming_train_edges.tolist()
        else:  # TODO 在ref中，这里相当于删掉了节点u的所有入边，但是它的出边仍然保留。但是无向图，这种情况要求对所有出边也进行删除。
            if uniform:
                raise ValueError("uniform should not have dropped users! something wrong!")
            dropped_count += 1
            dropped_users.append(u)  # 如果节点u的degree超过阈值

    print('dropped count', dropped_count)
    print("dropped nodes", dropped_users)
    sampled_edges = reverse_edges(sampled_reversed_edges)

    # For non-train nodes, we can sample the entire edgelist.
    for u in all_nodes:
        if u not in train_nodes:
            sampled_edges[u] = edges[u]
    return sampled_edges


def sample_adjacency_lists(edges, train_nodes,
                           max_degree, uniform=False):
    """Statelessly samples the adjacency lists with in-degree constraints.

    This implementation performs Bernoulli sampling over edges.

    Note that the degree constraint only applies to training subgraphs.
    The validation and test subgraphs are sampled completely.

    Args:
      edges: The adjacency lists to sample.
      train_nodes: A sequence of train nodes.
      max_degree: The bound on in-degree for any node over training subgraphs.
      rng: The PRNGKey for reproducibility
    Returns:
      A sampled adjacency list, indexed by nodes.
    """
    train_nodes = set(train_nodes)
    all_nodes = edges.keys()

    reversed_edges = reverse_edges(edges)  # 逆邻接表记录每个节点的入边节点
    sampled_reversed_edges = {u: [] for u in all_nodes}

    # For every node, bound the number of incoming edges from training nodes.
    dropped_count = 0
    dropped_users = []
    for u in all_nodes:
        # u_rng = jax.random.fold_in(rng, u)
        incoming_edges = reversed_edges[u]
        incoming_train_edges = [v for v in incoming_edges if v in train_nodes]
        if not incoming_train_edges:  # 只处理入边中训练集中的节点
            continue

        in_degree = len(incoming_train_edges)
        if not uniform:
            sampling_prob = max_degree / (2 * in_degree)
            # sampling_mask = (
            #         jax.random.uniform(u_rng, shape=(in_degree,)) <= sampling_prob)
            sampling_mask = (
                    np.random.uniform(size=(in_degree,)) <= sampling_prob)
            sampling_mask = np.asarray(sampling_mask)
        else:  # 取出max degree个边
            sampling_mask = np.array([False] * in_degree)
            sample_size = min(in_degree, max_degree)
            inds = np.random.choice(np.arange(len(sampling_mask)), size=sample_size, replace=False)
            sampling_mask[inds] = True
        incoming_train_edges = np.asarray(incoming_train_edges)[sampling_mask]
        unique_incoming_train_edges = np.unique(incoming_train_edges)

        # Check that in-degree is bounded, otherwise drop this node.
        if len(unique_incoming_train_edges) <= max_degree:
            sampled_reversed_edges[u] = unique_incoming_train_edges.tolist()
        else:
            dropped_count += 1
            dropped_users.append(u)

    print('dropped count', dropped_count)
    print("dropped nodes", dropped_users)
    sampled_edges = reverse_edges(sampled_reversed_edges)

    # For non-train nodes, we can sample the entire edgelist.
    for u in all_nodes:
        if u not in train_nodes:
            sampled_edges[u] = edges[u]
    return sampled_edges


def edge_cheking_utils(graph: torch.Tensor):
    # The graph shape = (2, N)
    graphT = graph.T
    graphT = tuple(map(tuple, graphT.numpy()))
    from collections import Counter
    replicated_item = {}
    tmp = dict(Counter(graphT))
    for key, value in tmp.items():
        if value > 1:
            replicated_item[key] = value
    sampled_out_edges = []
    graph_set = set(graphT)  # filter out the replicated edges
    return graph_set


def degree_constrained_check(graph: torch.Tensor, max_node_degree: int, type="outgoing"):
    edges = get_adjacency_lists_from_tensor(graph)
    violated_node = {}
    if type == "incoming":
        edges = reverse_edges(edges)
    for u, neighbors in edges.items():
        if len(neighbors) > max_node_degree:
            violated_node[u] = neighbors
    return violated_node


def subsample_graph(graph: [Data, torch.Tensor], max_degree, uniform=False, undirected=False):
    if isinstance(graph, Data):
        edges = get_adjacency_lists(graph)
        if hasattr(graph, "train_mask"):
            train_indices = torch.where(graph.train_mask)[0].tolist()
        else:
            train_indices = graph.edge_index.unique().tolist()
    elif isinstance(graph, torch.Tensor):
        edges = get_adjacency_lists_from_tensor(graph)
        train_indices = graph.unique().tolist()
    else:
        raise ValueError("Invalid graph type, must be pyg Data or Tensor")
    if undirected:
        edges = sample_adjacency_lists_undirected_graph(edges, train_indices, max_degree, uniform=uniform)
    else:
        edges = sample_adjacency_lists(edges, train_indices, max_degree, uniform=uniform)

    senders = []
    receivers = []
    for u in edges:
        for v in edges[u]:
            senders.append(u)
            receivers.append(v)
    edge_index = torch.tensor([senders, receivers])
    if isinstance(graph, Data):
        graph.edge_index = edge_index
    elif isinstance(graph, torch.Tensor):
        graph = edge_index
    return graph


def filter_out_one_way_edges(edge_index: torch.Tensor):
    edge_index_list = edge_index.T.tolist()
    filtered_edges = [edge for edge in edge_index_list if edge[::-1] in edge_index_list]
    return torch.tensor(filtered_edges).T


def subsample_graph_for_undirected_graph_old(graph, max_degree):
    # Graph should be the upper triangle elements of adjacency matrix
    graph = subsample_graph(graph, max_degree, uniform=True)
    graph = graph[[-1, 0]]
    graph = subsample_graph(graph, max_degree, uniform=True)
    graph = graph[[-1, 0]]
    return graph


def subsample_graph_for_undirected_graph(graph, max_degree):
    undirected = True
    from torch_geometric.utils import to_undirected
    if isinstance(graph, torch.Tensor):
        graph = to_undirected(graph)
        # temp_graph = ssp.csc_matrix((np.ones(graph.shape[1]), (graph[0].numpy(), graph[1].numpy())))
        # temp_graph = temp_graph.toarray()
    if undirected:
        graph = subsample_graph(graph, max_degree, uniform=True, undirected=undirected)
        from torch_geometric.utils import is_undirected
        assert is_undirected(graph), "sampled graph is not undirected, something wrong!"
    else:
        graph = subsample_graph(graph, max_degree, uniform=True)
        graph = filter_out_one_way_edges(graph)
    graph = ssp.csc_matrix((np.ones(graph.shape[1]), (graph[0].numpy(), graph[1].numpy())))
    graph_triu = ssp.triu(graph, k=1)
    row, col, _ = ssp.find(graph_triu)
    graph = torch.tensor([row, col], dtype=torch.int)
    return graph

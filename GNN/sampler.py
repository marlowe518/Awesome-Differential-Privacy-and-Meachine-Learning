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


def sample_adjacency_lists(edges, train_nodes,
                           max_degree):
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
        sampling_prob = max_degree / (2 * in_degree)
        # sampling_mask = (
        #         jax.random.uniform(u_rng, shape=(in_degree,)) <= sampling_prob)
        sampling_mask = (
                np.random.uniform(size=(in_degree,)) <= sampling_prob)
        sampling_mask = np.asarray(sampling_mask)

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


def subsample_graph(graph: [Data, torch.Tensor], max_degree):
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
    edges = sample_adjacency_lists(edges, train_indices, max_degree)
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


def subsample_graph_for_undirected_graph(graph, max_degree):
    from torch_geometric.utils import to_undirected
    if isinstance(graph, torch.Tensor):
        graph = to_undirected(graph)
    graph = subsample_graph(graph, max_degree)
    # graph = filter_out_one_way_edges(graph)
    graph = ssp.csc_matrix((np.ones(graph.shape[1]), (graph[0].numpy(), graph[1].numpy())))
    graph_triu = ssp.triu(graph, k=1)
    graph_tril = ssp.tril(graph, k=1).T
    graph_triu_filtered = graph_triu.multiply(graph_tril)  # Point-wise multiply to eliminate unsymmetrical element.
    row, col, _ = ssp.find(graph_triu_filtered)
    graph = torch.tensor([row, col], dtype=torch.int)
    return graph

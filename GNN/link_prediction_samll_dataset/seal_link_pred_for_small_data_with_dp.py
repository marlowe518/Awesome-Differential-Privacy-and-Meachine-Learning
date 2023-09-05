import argparse
import math
import time
import os, sys
import os.path as osp

sys.path.append("D:\opensource\jeff\Awesome-Differential-Privacy-and-Meachine-Learning")
from shutil import copy
import copy as cp

import scipy
from tqdm import tqdm
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from sklearn.metrics import roc_auc_score

import warnings

from optimizer.dp_optimizer import DPSGD

warnings.filterwarnings("ignore", category=UserWarning)

from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
# from models import *
from GNN.link_prediction.models import *
from GNN.sampler import subsample_graph, subsample_graph_for_undirected_graph

parser = argparse.ArgumentParser(description='SEAL_for_small_dataset')
# Dataset Setting
# parser.add_argument('--data_name', type=str, default="Yeast")
# parser.add_argument('--data_name', type=str, default="Router")
# parser.add_argument('--data_name', type=str, default="USAir")
# parser.add_argument('--data_name', type=str, default="Yeast")
parser.add_argument('--data_name', type=str, default="NS")
# parser.add_argument('--data_name', type=str, default="PB")
# parser.add_argument('--data_name', type=str, default="Ecoli")

parser.add_argument('--uniq_appendix', type=str, default="_20230904")

# Subgraph extraction settings
parser.add_argument('--node_label', type=str, default='drnl',
                    help="which specific labeling trick to use")
parser.add_argument('--num_hops', type=int, default=2,
                    help="num_hops is the path length in path subgraph while in neighborhood it is the radius of neighborhood")
parser.add_argument('--use_feature', default=False,
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', default=None)
parser.add_argument('--max_node_degree', type=int, default=40)
parser.add_argument('--check_degree_constrained', default=False)
parser.add_argument('--check_degree_distribution', default=False)
parser.add_argument('--neighborhood_subgraph', action='store_true')

# GNN Setting
parser.add_argument('--model', type=str, default="GCN")
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)

parser.add_argument('--dynamic_train', action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--eval_metric', default="auc")
parser.add_argument('--hitsK', default=50)

# Training settings
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--micro_batch', type=bool, default=True,
                    help="non-dp training with microbatch")  # Remember any args passed from command line is strin, the following bool params only for manually tune
parser.add_argument('--dp_no_noise', type=bool, default=False, help="dp training without noise")

# Privacy settings
parser.add_argument('--random_seed', type=int, default=1234)
parser.add_argument('--dp_method', type=str, default="DPLP")
parser.add_argument('--target_epsilon', type=float, default=None)
parser.add_argument('--lets_dp', type=bool, default=True)
parser.add_argument('--max_norm', type=float, default=100.)
parser.add_argument('--sigma', type=float, default=0.01)
parser.add_argument('--target_delta', type=float, default=1e-5)

# Testing settings
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='',
                    help="an appendix to the data directory")  # The processed data saving path
parser.add_argument('--save_appendix', type=str, default='',
                    help="an appendix to the save directory")  # The log path
parser.add_argument('--keep_old', action='store_true',
                    help="do not overwrite old files in the save directory")  # save the running script

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_name):
    """
    :param data_name:
    :return:
    A_csc: csc类型（按列压缩的稀疏矩阵），这个矩阵是对称的
    """
    split_edge = torch.load(f"./data/processed_data/{data_name}_split_edges")
    import scipy.io as sio
    mat_data = sio.loadmat(f"./data/processed_data/{data_name}_train.mat")
    A_csc = mat_data["net"]
    return A_csc, split_edge


def inspect_data_frequency_distribution(values, bin_nums=10, title=""):
    bins = pd.cut(values, bins=bin_nums)
    print("bin_counts\n", bins.value_counts())
    plt.hist(values, bins=bin_nums)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("counts")
    plt.show()


class SEALDatasetSmall(InMemoryDataset):
    def __init__(self, root, A_csc, split_edge, num_hops, node_features=None, percent=100,
                 split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False):
        """
        :param root:
        :param A_csc: csc格式，仅包含训练数据的对称邻接矩阵。注意：包含训练数据和包含训练数据中有标签的数据在这里是截然不同的意思.
                    A_csc矩阵的作用是提取子图，需要提取的边参照A_csc,而不是pos_edge。后者表示的是正样本数量。
        :param split_edge:
        :param num_hops:
        :param node_features:
        :param percent:
        :param split:
        :param use_coalesce:
        :param node_label:
        :param ratio_per_hop:
        :param max_nodes_per_hop:
        :param directed:
        """
        self.A_csc: ssp.csr_matrix = A_csc
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.node_features = node_features
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.process_log_path = root + "/log"
        self.neighborhood_subgraph = args.neighborhood_subgraph  # TODO as input parameter
        self.dp_method = args.dp_method
        self.max_node_degree = args.max_node_degree
        self.target_epsilon = args.target_epsilon
        self.target_delta = args.target_delta
        self.random_seed = args.random_seed
        self.noise_type = "gaussian"
        super(SEALDatasetSmall, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        # Here we do not sample pos edges and neg edges should be already included.
        assert "edge_neg" in self.split_edge[self.split].keys(), "neg edges must be given"
        pos_edge = self.split_edge[self.split]["edge"].t()
        neg_edge = self.split_edge[self.split]["edge_neg"].t()
        with open(self.process_log_path, 'a') as f:
            print(f">>>>>>>>Processing {self.split} edges>>>>>>>>", file=f)
            print(f"pos_edge nums:{pos_edge.shape[1]}", file=f)
            print(f"neg_edge nums:{neg_edge.shape[1]}", file=f)

        # Here we sample the positive and negative edges to meet the constraint of node degree.

        if self.split == "train":
            key_results["original_edges"] = pos_edge.shape[1]
            # TODO: In the future, the pos edges can be sampled, while the train indices remain whole training graph (i.e., A_csc)
            pos_edge_degree_constrained: torch.Tensor = subsample_graph_for_undirected_graph(pos_edge,
                                                                                             max_degree=args.max_node_degree)
            neg_edge_degree_constrained: torch.Tensor = subsample_graph_for_undirected_graph(neg_edge,
                                                                                             max_degree=args.max_node_degree)
            # Shuffle and ensure #pos_edges = #neg_edges
            indexes = np.arange(min(pos_edge_degree_constrained.shape[1], neg_edge_degree_constrained.shape[1]))
            random.shuffle(indexes)
            pos_edge_degree_constrained = pos_edge_degree_constrained[:, indexes]
            neg_edge_degree_constrained = neg_edge_degree_constrained[:, indexes]
            degree_constrained_csc_mat = ssp.csc_matrix((np.ones(len(pos_edge_degree_constrained[0])),
                                                         (pos_edge_degree_constrained[0],
                                                          pos_edge_degree_constrained[1])),
                                                        shape=self.A_csc.shape, dtype=int)
            degree_constrained_csc_mat = degree_constrained_csc_mat + degree_constrained_csc_mat.T - ssp.diags(
                degree_constrained_csc_mat.diagonal(), dtype=int)  # To undirected graph
            if self.dp_method == "LapGraph":
                from linkteller import perturb_adj_continuous
                degree_constrained_csc_mat = perturb_adj_continuous(degree_constrained_csc_mat, noise_type="gaussian",
                                                                    target_epsilon=self.target_epsilon,
                                                                    target_delta=self.target_delta,
                                                                    noise_seed=self.random_seed)
            self.A_csc = degree_constrained_csc_mat
            assert scipy.linalg.issymmetric(self.A_csc.toarray()), "Train_net must be symmetric!"
            pos_edge = pos_edge_degree_constrained
            neg_edge = neg_edge_degree_constrained
            key_results["sampled_edges"] = pos_edge.shape[1]
            with open(self.process_log_path, 'a') as f:
                print("After edge sampling", file=f)
                print(f"pos_edge nums:{pos_edge.shape[1]}", file=f)
                print(f"neg_edge nums:{neg_edge.shape[1]}", file=f)

            if args.check_degree_constrained:
                from GNN.sampler import degree_constrained_check
                pos_edge_degree_unlimited_nodes_outgoing = degree_constrained_check(pos_edge_degree_constrained,
                                                                                    args.max_node_degree)
                neg_edge_degree_unlimited_nodes_outgoing = degree_constrained_check(neg_edge_degree_constrained,
                                                                                    args.max_node_degree)
                pos_edge_degree_unlimited_nodes_incoming = degree_constrained_check(pos_edge_degree_constrained,
                                                                                    args.max_node_degree,
                                                                                    type="incoming")
                neg_edge_degree_unlimited_nodes_incoming = degree_constrained_check(neg_edge_degree_constrained,
                                                                                    args.max_node_degree,
                                                                                    type="incoming")
                A_arr = self.A_csc.toarray()
                degree_distribution = A_arr.sum(axis=1)
                violated_nodes = np.where(degree_distribution > args.max_node_degree)
                assert not violated_nodes, f"Nodes:{violated_nodes} does not meet the degree constraint!"

            if args.check_degree_distribution:
                import matplotlib.pyplot as plt
                import pandas as pd
                out_degree_distribution = np.array(self.A_csc.sum(axis=1)).reshape(-1)
                in_degree_distribution = np.array(self.A_csc.sum(axis=0)).reshape(-1)
                df = pd.DataFrame(out_degree_distribution, columns=["out_degree"])
                df["in_degree"] = in_degree_distribution
                # df["binned"] = pd.cut(degree_distribution, bins=10)
                with open(self.process_log_path, 'a') as f:
                    print(f"Out degree distribution:", file=f)
                    print(pd.value_counts(df["out_degree"]), file=f)
                    print(f"In degree distribution:", file=f)
                    print(pd.value_counts(df["in_degree"]), file=f)
                    # print(pd.value_counts(df["binned"]), file=f)
                # plt.hist(in_degree_distribution, bins=len(df["in_degree"].unique()))
                # plt.xlabel("Degree")
                # plt.ylabel("Frequency")
                # plt.show()

        # ------------------------------------------------------------- #
        A_csr = self.A_csc.tocsr()

        if not self.directed:
            self.A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A_csr, self.node_features, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, self.A_csc, self.neighborhood_subgraph)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A_csr, self.node_features, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, self.A_csc, self.neighborhood_subgraph)
        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


# def compute_max_terms_per_edge(num_message_passing_steps, max_node_degree):
#     max_node_degree = 2 * max_node_degree ** 2
#     max_terms_per_edge = sum([max_node_degree ** i for i in range(1, num_message_passing_steps + 1)])
#     # max_terms_per_edge = min(max_terms_per_edge, args.batch_size)  # Bounded by batch_size
#     return max_terms_per_edge

def compute_max_terms_per_edge_for_neighborhood(num_message_passing_steps, max_node_degree, lamda=1):
    number_of_low_hop_neighbors = 0
    for i in range(num_message_passing_steps):
        number_of_low_hop_neighbors += 2 * max_node_degree ** i
    number_of_h_hop_neighbors = max_node_degree ** num_message_passing_steps
    max_terms_per_edge = (number_of_low_hop_neighbors + number_of_h_hop_neighbors) * (1 + lamda) * max_node_degree
    return max_terms_per_edge


def compute_max_terms_per_edge(num_message_passing_steps, max_node_degree, num_hops):
    if args.neighborhood_subgraph:
        return compute_max_terms_per_edge_for_neighborhood(num_message_passing_steps, max_node_degree)
    else:
        return compute_max_terms_per_edge_for_path(num_hops, max_node_degree)


def compute_max_terms_per_edge_for_path_deprecated(path_length, max_node_degree, lamda=1):
    hop = math.floor((path_length - 1) / 2)
    max_terms = 0
    for i in range(hop):
        max_terms += 2 * lamda * max_node_degree ** (i + 1)
    tail_node_num = min((1 + lamda) * max_node_degree, max_node_degree ** hop)
    head_node_num = max_node_degree ** hop
    max_terms += head_node_num * tail_node_num
    return max_terms


def compute_max_terms_per_edge_for_path(path_length, max_node_degree, lamda=1):
    if path_length > 2:
        hop = math.ceil((path_length - 1) / 2)
        r = (path_length - 1) % 2
        implicated_edges_incident_to_lower_node = 0
        for i in range(hop - r):  # i=0,...,h-1-r
            tail_node_num_of_q = 0
            for j in range(i, 2 * hop):
                tail_node_num_of_q += max_node_degree ** j
            q_higher_order_neighbors_num = min(lamda * max_node_degree, tail_node_num_of_q)
            tail_node_num_of_s = tail_node_num_of_q - max_node_degree ** i
            s_higher_order_neighbors_num = min(lamda * max_node_degree, tail_node_num_of_s)
            implicated_edges_incident_to_lower_node += max_node_degree ** i * (
                    q_higher_order_neighbors_num + s_higher_order_neighbors_num)
        tmp_1 = min((1 + lamda) * max_node_degree, max_node_degree ** hop)
        implicated_edges_incident_to_high_order_node = (1 + r) * max_node_degree ** (hop - r) * tmp_1
        edges_on_rings = 0
        for i in range(1, hop - r + 1):
            edges_on_rings += 2 * min(max_node_degree ** (hop + 1), max_node_degree ** i)
        all_implicated_edges = implicated_edges_incident_to_lower_node \
                               + implicated_edges_incident_to_high_order_node \
                               + edges_on_rings
    elif path_length == 2:
        all_implicated_edges = 2 * max_node_degree
    else:
        raise ValueError(f"not a valid path_length of {path_length}")
    return all_implicated_edges


def parameter_selection_loss(path_length, max_node_degree, A_csc, split_edge,
                             epsilon, beta_1, beta_2, gamma=0.001, lamda=1, differentially_private=False):
    # Only applicable to path subgraph
    max_terms = compute_max_terms_per_edge_for_path(path_length, max_node_degree, lamda)
    pos_edge_nums = split_edge["train"]["edge"].shape[0]
    per_node_out_degree_distribution = np.array(A_csc.sum(axis=1)).reshape(-1)
    per_node_in_degree_distribution = np.array(A_csc.sum(axis=0)).reshape(-1)
    df = pd.DataFrame(per_node_out_degree_distribution, columns=["out_degree"])
    df["in_degree"] = per_node_in_degree_distribution
    out_degree_distribution = pd.value_counts(df["out_degree"])
    # assert per_node_out_degree_distribution == per_node_in_degree_distribution, "out degree does not equal to in degree"
    node_over_max_node_degree_num = np.where(per_node_out_degree_distribution > max_node_degree)[0].size
    max_degree = np.max(per_node_out_degree_distribution)
    if differentially_private:
        pos_edge_nums += np.random.laplace(0, 1 / (epsilon * beta_1))
        node_over_max_node_degree_num += np.random.laplace(0, 2 / (epsilon * beta_2))
        max_degree += np.random.laplace(0, 1 / epsilon * (1 - beta_1 - beta_2))
    effective_sample_nums = (1 - gamma ** path_length) * (1 + lamda) * (
            pos_edge_nums - node_over_max_node_degree_num * ((max_degree - max_node_degree) / 2))
    effective_sample_nums = round(effective_sample_nums)
    loss = max_terms / effective_sample_nums
    return loss


def compute_base_sensitivity(num_message_passing_steps, max_degree, batch_size, num_hops, dp_method):
    """Returns the base sensitivity which is multiplied to the clipping threshold.

    Args:

    """
    if dp_method == "DPLP":
        max_terms_per_edge = compute_max_terms_per_edge(num_message_passing_steps, max_degree, num_hops=num_hops)
        max_terms_per_edge = min(max_terms_per_edge, batch_size)
    elif dp_method == "DPGNN4GC":
        max_terms_per_edge = batch_size
    return float(2 * max_terms_per_edge)


def train_dynamic_add_noise(model, train_loader, optimizer, criterion, full_batch=False):
    '''
    Args:
        model:
        train_loader: PyG DataLoader
        optimizer:

    Returns:

    '''
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0

    for id, data in enumerate(train_loader):  # TODO per-sample computation for mini-batch
        optimizer.zero_accum_grad()  # 梯度清空
        for id in range(data.num_graphs):
            data_microbatch = data[id]
            data_microbatch.to(device)
            optimizer.zero_microbatch_grad()
            x = data_microbatch.x if args.use_feature else None
            edge_weight = data_microbatch.edge_weight if args.use_edge_weight else None
            node_id = data_microbatch.node_id if emb else None
            logits = model(data_microbatch.z, data_microbatch.edge_index, data_microbatch.batch, x, edge_weight,
                           node_id, micro_batch=True)
            # loss = criterion(logits.view(-1), data_microbatch.y.to(torch.float))
            loss = BCEWithLogitsLoss()(logits.view(-1), data_microbatch.y.to(torch.float))
            loss.backward()  # 梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
            train_loss += loss.item()
        if args.dp_no_noise:
            optimizer.step_dp(no_noise=True)  # 这个做的是梯度加噪和梯度平均更新下降的操作 TODO Remove the "no_noise" attribute
        else:
            optimizer.step_dp()
    return train_loss / len(train_loader.dataset), train_acc  # 返回平均损失和平均准确率


def train(model, train_loader, optimizer, train_dataset, emb=None):
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        if args.micro_batch:
            break
    return total_loss / len(train_dataset)


def account_privacy_dpsgd(step_num, target_delta, sigma, orders):
    from privacy_analysis.RDP.rdp_convert_dp import compute_eps
    from privacy_analysis.RDP.compute_rdp import compute_rdp
    # Note that the step_num is the accumulating step.
    rdp = compute_rdp(1., sigma, step_num, orders)
    epsilon, best_alpha = compute_eps(orders, rdp, target_delta)
    return epsilon, best_alpha


def account_privacy(num_message_passing_steps,
                    max_node_degree,
                    num_hops,
                    step_num,
                    batch_size,
                    train_num,
                    sigma,
                    orders,
                    target_delta=1e-5,
                    ):
    from privacy_analysis.RDP.compute_multiterm_rdp import compute_multiterm_rdp
    from privacy_analysis.RDP.rdp_convert_dp import compute_eps

    max_terms_per_edge = compute_max_terms_per_edge(num_message_passing_steps,
                                                    max_node_degree,
                                                    num_hops)
    max_terms_per_edge = min(max_terms_per_edge, train_num)
    # assert max_terms_per_node <= len(train_dataset), "#affected terms must <= #samples"
    rdp_every_epoch = compute_multiterm_rdp(orders, step_num, sigma, train_num,
                                            max_terms_per_edge, batch_size)
    # rdp_every_epoch_org = compute_rdp(args.batch_size / len(train_dataset), args.sigma, 1 * epoch, orders)
    epsilon, best_alpha = compute_eps(orders, rdp_every_epoch, target_delta)
    return epsilon, best_alpha


def get_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        step_num: int,
        orders: list,
        num_message_passing_steps,
        max_node_degree,
        num_hops,
        batch_size,
        train_num,
        epsilon_tolerance: float = 0.01,
        dp_method="DPLP"
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 1000  # 从0-10进行搜索，一般的sigma设置也不会超过这个范围。其实从0-5就可以了我觉得。TODO 这个数过大可能导致rdp<0
    if dp_method == "DPGNN4GC":
        eps_high, best_alpha = account_privacy_dpsgd(step_num=step_num, target_delta=args.target_delta,
                                                     sigma=args.sigma, orders=orders)
    else:
        eps_high, best_alpha = account_privacy(num_message_passing_steps=num_message_passing_steps,
                                               max_node_degree=max_node_degree,
                                               num_hops=num_hops,
                                               step_num=step_num,
                                               batch_size=batch_size,
                                               train_num=train_num,
                                               sigma=sigma_high,
                                               target_delta=target_delta,
                                               orders=orders)

    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    # 下面是折半搜索，直到找到满足这个eps容忍度的sigma_high,sigma是从大到小搜索，即eps从小到大逼近
    while target_epsilon - eps_high > epsilon_tolerance:  # 我们希望当目前eps减去当前计算出来的eps小于容忍度，也就是计算出来的eps非常接近于目标eps
        sigma = (sigma_low + sigma_high) / 2
        if dp_method == "DPGNN4GC":
            eps, best_alpha = account_privacy_dpsgd(step_num=step_num, target_delta=args.target_delta,
                                                    sigma=args.sigma, orders=orders)
        else:
            eps, best_alpha = account_privacy(num_message_passing_steps=num_message_passing_steps,
                                              max_node_degree=max_node_degree,
                                              num_hops=num_hops,
                                              step_num=step_num,
                                              batch_size=batch_size,
                                              train_num=train_num,
                                              sigma=sigma,
                                              target_delta=target_delta,
                                              orders=orders)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return round(sigma_high, 2)


def train_with_dp(model, optimizer, train_dataset, epoch, emb=None, dp_method="DPLP"):
    model.train()
    criterion = BCEWithLogitsLoss()
    indices = np.random.choice(range(len(train_dataset)), size=(args.batch_size,), replace=False)
    train_batch_subgraphs = [train_dataset[i] for i in indices]
    train_loader = DataLoader(train_batch_subgraphs, batch_size=args.batch_size, num_workers=0, shuffle=False)
    train_loss, train_acc = train_dynamic_add_noise(model, train_loader, optimizer, criterion,
                                                    full_batch=False)
    print(f"epoch:{epoch}, total loss:{train_loss}")

    # ------------------- privacy accounting ------------------- #
    orders = np.arange(1, 10, 0.1)[1:]
    if dp_method == "DPGNN4GC":
        epsilon, best_alpha = account_privacy_dpsgd(step_num=epoch, target_delta=args.target_delta,
                                                    sigma=args.sigma, orders=orders)
    else:
        epsilon, best_alpha = account_privacy(num_message_passing_steps=args.num_layers,
                                              max_node_degree=args.max_node_degree,
                                              num_hops=args.num_hops,
                                              step_num=epoch,
                                              batch_size=args.batch_size,
                                              train_num=len(train_dataset),
                                              sigma=args.sigma,
                                              orders=orders,
                                              target_delta=args.target_delta)
    # from privacy_analysis.RDP.compute_multiterm_rdp import compute_multiterm_rdp
    # from privacy_analysis.RDP.rdp_convert_dp import compute_eps
    # orders = np.arange(1, 10, 0.1)[1:]
    # max_terms_per_edge = compute_max_terms_per_edge(num_message_passing_steps=args.num_layers,
    #                                                 max_node_degree=args.max_node_degree,
    #                                                 num_hops=args.num_hops)
    # max_terms_per_edge = min(max_terms_per_edge, len(train_dataset))
    # # assert max_terms_per_node <= len(train_dataset), "#affected terms must <= #samples"
    # rdp_every_epoch = compute_multiterm_rdp(orders, epoch, args.sigma, len(train_dataset),
    #                                         max_terms_per_edge, args.batch_size)
    #
    # from privacy_analysis.RDP.compute_rdp import compute_rdp
    # rdp_every_epoch_org = compute_rdp(args.batch_size / len(train_dataset), args.sigma, 1 * epoch, orders)
    # epsilon, best_alpha = compute_eps(orders, rdp_every_epoch, args.target_delta)
    # # epsilon_org, best_alpha_org = compute_eps(orders, rdp_every_epoch_org, args.target_delta)
    print("epoch: {:3.0f}".format(epoch) + " | epsilon: {:10.7f}".format(
        epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))
    # print("epoch: {:3.0f}".format(epoch) + " | epsilon_org: {:10.7f}".format(
    #     epsilon_org) + " | best_alpha: {:7.4f}".format(best_alpha_org))
    return train_loss, epsilon


def eval_hits(y_pred_pos, y_pred_neg, K, type_info="torch"):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    if type_info == 'torch':
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    # type_info is numpy
    else:
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

    return {'hits@{}'.format(K): hitsK}


def plot_roc_curve(true, pred):
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(true, pred)
    plt.plot(fpr, tpr)
    plt.show()


@torch.no_grad()
def test(model, val_loader, test_loader, emb=None):
    model.eval()
    y_pred, y_true = [], []
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true == 1]
    neg_val_pred = val_pred[val_true == 0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]
    if args.eval_metric == 'hits':
        results = {}
        valid_hits = eval_hits(pos_val_pred, neg_val_pred, args.hitsK)[f'hits@{args.hitsK}']
        test_hits = eval_hits(pos_test_pred, neg_test_pred, args.hitsK)[f'hits@{args.hitsK}']
        results[f'Hits@{args.hitsK}'] = (valid_hits, test_hits)
    elif args.eval_metric == "auc":
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)
    else:
        raise ValueError("Invalid eval metric!")
    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results


def main():
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    A_csc, split_edge = load_data(args.data_name)
    if not args.neighborhood_subgraph:
        loss_indicator = parameter_selection_loss(args.num_hops, args.max_node_degree, A_csc, split_edge, epsilon=1,
                                                  beta_1=0.3, beta_2=0.3,
                                                  differentially_private=False)
    else:
        loss_indicator = 0.
    train_dataset: SEALDatasetSmall = SEALDatasetSmall(dataset_path, A_csc, split_edge, args.num_hops,
                                                       node_features=None,
                                                       percent=args.train_percent, split="train", directed=directed)
    val_nums = int(val_ratio * len(train_dataset))
    train_dataset = train_dataset.shuffle()
    val_dataset: SEALDatasetSmall = train_dataset[:val_nums]
    train_dataset = train_dataset[val_nums:]
    test_dataset: SEALDatasetSmall = SEALDatasetSmall(dataset_path, A_csc, split_edge, args.num_hops,
                                                      node_features=None,
                                                      percent=args.test_percent, split="test", directed=directed)
    test_dataset = test_dataset.shuffle()  # TODO
    args.batch_size = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    if args.target_epsilon and (args.dp_method in ["DPLP", "DPGNN4GC"]):
        orders = np.arange(1, 10, 0.1)[1:]
        sigma = get_noise_multiplier(args.target_epsilon,
                                     args.target_delta,
                                     num_message_passing_steps=args.num_layers,
                                     max_node_degree=args.max_node_degree,
                                     num_hops=args.num_hops,
                                     step_num=args.epochs,
                                     batch_size=args.batch_size,
                                     train_num=len(train_dataset),
                                     orders=orders,
                                     dp_method=args.dp_method)
        args.sigma = sigma
        print(f"Given target epsilon, sigma is set to:{args.sigma}")

    for run in range(args.runs):
        # model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
        #               train_dataset, args.dynamic_train, use_feature=args.use_feature,
        #               node_embedding=emb).to(device)
        if args.model == "GCN":
            model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, use_feature=args.use_feature,
                        node_embedding=emb).to(device)
        else:
            raise ValueError(f"{args.model} model is not supported!")
        if torch.cude.device_count() > 1:
            import torch.nn as nn
            print(f"{torch.cude.device_count()} GPUs are used")
            model = nn.DataParallel(model)
        parameters = list(model.parameters())
        if args.lets_dp:
            sens = compute_base_sensitivity(max_degree=args.max_node_degree,
                                            num_message_passing_steps=args.num_layers,
                                            num_hops=args.num_hops, batch_size=args.batch_size,
                                            dp_method=args.dp_method)
            optimizer = DPSGD(
                l2_norm_clip=args.max_norm,  # 裁剪范数
                noise_multiplier=args.sigma * sens,
                minibatch_size=args.batch_size,  # 几个样本梯度进行一次梯度下降
                microbatch_size=1,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
                params=model.parameters(),
                lr=args.lr,
                momentum=args.momentum
            )
            print(f"Number of train samples:{len(train_dataset)}")
            print(
                f"Max term per edges:{compute_max_terms_per_edge(args.num_layers, args.max_node_degree, args.num_hops)}")
            print(f"Sens:{sens}")
            with open(log_file, 'a') as f:
                print(f"Number of train samples:{len(train_dataset)}", file=f)
                print(
                    f"Max term per edges:{compute_max_terms_per_edge(args.num_layers, args.max_node_degree, args.num_hops)}",
                    file=f)
                print(f"Sens:{sens}", file=f)
        else:
            optimizer = torch.optim.SGD(params=parameters, lr=args.lr, momentum=args.momentum)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')

        # Save model parameters descriptions into log file
        with open(log_file, 'a') as f:
            print(f'Total number of parameters is {total_params}', file=f)

        # Training
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + args.epochs):
            if args.lets_dp:
                loss, eps = train_with_dp(model, optimizer, train_dataset, epoch, emb, dp_method=args.dp_method)
            else:
                loss = train(model, train_loader, optimizer, train_dataset, emb)
            if epoch % args.eval_steps == 0:
                results = test(model, val_loader, test_loader, emb)
                if args.lets_dp:
                    results["EPS"] = (eps, eps)
                # Record the results to logger
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run:02d}, Epoch: {epoch:02d}, ' +
                                    f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                    f'Test: {100 * test_res:.2f}%')
                        print("\n" + key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)
        # Recording the
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f)
        if "AUC" in loggers:
            argmax = loggers["AUC"].return_statistics(run)
            val_res = loggers["AUC"].results[run][argmax][0] * 100
            test_res = loggers["AUC"].results[run][argmax][1] * 100
            final_val_res = loggers["AUC"].results[run][-1][0] * 100
            final_test_res = loggers["AUC"].results[run][-1][1] * 100
            key_results["all_runs"]["best_epoch"].append(argmax)
            key_results["all_runs"]["highest_val"].append(val_res)
            key_results["all_runs"]["final_test"].append(test_res)
            key_results["all_runs"]["final_round_val"].append(final_val_res)
            key_results["all_runs"]["final_round_test"].append(final_test_res)
            key_results["all_runs"]["val_test_trend"].append(loggers["AUC"].results[run])
            if "EPS" in loggers:
                eps_at_highest_val = float(loggers["EPS"].results[run][argmax][0])
                key_results["all_runs"]["eps"].append(eps_at_highest_val)
                key_results["all_runs"]["eps_trend"].append(loggers["EPS"].results[run])

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    print(f'Total number of parameters is {total_params}')
    print(f'Results are saved in {args.res_dir}')

    key_results["experiment"] = args.res_dir
    key_results["max_node_degree"] = args.max_node_degree
    key_results["num_hop"] = args.num_hops
    if args.lets_dp:
        key_results["max_term_per_edge"] = compute_max_terms_per_edge(max_node_degree=args.max_node_degree,
                                                                      num_message_passing_steps=args.num_layers,
                                                                      num_hops=args.num_hops)
        key_results["sens"] = compute_base_sensitivity(max_degree=args.max_node_degree,
                                                       num_message_passing_steps=args.num_layers,
                                                       num_hops=args.num_hops,
                                                       batch_size=args.batch_size,
                                                       dp_method=args.dp_method)
        key_results["parameter_indicator"] = loss_indicator
        key_results["eps"] = np.mean(key_results["all_runs"]["eps"])

    key_results["lr"] = args.lr
    key_results["sigma"] = args.sigma
    key_results["max_norm"] = args.max_norm
    key_results["batch_size"] = args.batch_size
    key_results["dataset"] = args.data_name
    key_results["train_samples"] = len(train_dataset)
    key_results["best_epoch"] = np.mean(key_results["all_runs"]["best_epoch"])
    key_results["epsilon"] = args.target_epsilon
    key_results["highest_val"] = np.mean(key_results["all_runs"]["highest_val"])
    # key_results["val_std"] = np.round(np.std(key_results["all_runs"]["highest_val"]), 2) # TODO compared to
    key_results["val_std"] = torch.tensor(key_results["all_runs"]["highest_val"]).std().item()
    key_results["final_test"] = np.mean(key_results["all_runs"]["final_test"])
    # key_results["test_std"] = np.round(np.std(key_results["all_runs"]["final_test"]), 2)
    key_results["test_std"] = torch.tensor(key_results["all_runs"]["final_test"]).std().item()
    key_results["final_round_val"] = np.mean(key_results["all_runs"]["final_round_val"])
    key_results["final_round_val_std"] = torch.tensor(key_results["all_runs"]["final_round_val"]).std().item()
    key_results["final_round_test"] = np.mean(key_results["all_runs"]["final_round_test"])
    key_results["final_round_test_std"] = torch.tensor(key_results["all_runs"]["final_round_test"]).std().item()
    key_results["neighborhood_subgraph"] = str(args.neighborhood_subgraph)

    with open(log_file, 'a') as f:
        print(key_results, file=f)
    save_results(args.res_dir + "/key_results.pickle", key_results)


def save_results(file_path, obj):
    import pickle

    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
        print(f"results:{b}")


if __name__ == "__main__":
    # Check the correctness of parameters
    if not args.lets_dp:
        args.max_node_degree = 10000  # non-private method use all edges
    if args.dp_method == "LapGraph":
        args.max_node_degree = 10000
        args.neighborhood_subgraph = True
        args.sigma = np.nan
        args.lets_dp = False
        print(f"Method: {args.dp_method}, set max_node degree to 10000, and use neighborhood subgraph, "
              "and training without differential privacy")
    elif args.dp_method == "DPGNN4GC":
        args.max_node_degree = 10000
        args.neighborhood_subgraph = True
        print(f"Method: {args.dp_method}, set max_node degree to 10000, and use neighborhood subgraph, "
              "and training without differential privacy")
        # assert args.max_node_degree > 100, "for lap graph method, the max_node_degree should > 100"
        # assert args.neighborhood_subgraph == True, "for lap graph method, neighborhood_subgraph should be used"
    if not args.neighborhood_subgraph:  # not neighborhood subgraph
        assert args.num_layers >= math.floor(
            args.num_hops / 2), "num layers must >= maximum distance to ensure the training is based on path subgraph"
        # args.num_layers = math.floor((args.num_layers - 1) / 2)
        args.num_layers = args.num_hops
    else:
        args.num_layers = args.num_hops * 2 + 1
        # Key results
    key_results = dict.fromkeys(
        ["dataset", "experiment", "max_node_degree",
         "num_hop", "highest_val", "final_test", "val_std", "test_std", "best_epoch",
         "final_round_val", "final_round_val_std",
         "final_round_test", "final_round_test_std", "original_edges", "sampled_edges", "max_term_per_edge", "epsilon",
         "sigma"])
    key_results["all_runs"] = {key: [] for key in
                               ["highest_val", "final_test", "best_epoch", "val_test_trend", "final_round_val",
                                "final_round_test", "eps_trend", "eps"]}
    key_results["dp_method"] = args.dp_method
    # Create the path for saving data and log
    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y-%m-%d-%H-%M-%S")  # Mark the time
    if args.data_appendix == '':  # Create the data save path
        ns_ps_flag = "ns" if args.neighborhood_subgraph else "ps"
        args.data_appendix += '_d{}_{}_h{}_{}'.format(args.max_node_degree, ns_ps_flag, args.num_hops,
                                                      args.uniq_appendix)
        if args.dp_method == "LapGraph":
            args.data_appendix += '_{}_eps{}'.format(args.dp_method, args.target_epsilon)
    # Results include the log, running command, etc.
    args.res_dir = os.path.join('results/{}{}'.format(args.data_name, args.save_appendix))
    if not os.path.exists(args.res_dir):  # Create the result directory
        os.makedirs(args.res_dir)
    if not args.keep_old:  # Keep the running script or not
        # Backup python files.
        current_script_name = __file__.split("/")[-1]
        copy(current_script_name, args.res_dir)  # Save the running script to the results directory
        copy('utils.py', args.res_dir)  # The utils for data pre-processing
    log_file = os.path.join(args.res_dir, 'log.txt')  # Create the log file under res_dir
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:  # Save the command in cmd_input file
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')  # Save the command in log file
    with open(log_file, 'a') as f:
        f.write('\n' + cmd_input)
        print("Parameter setting:", file=f)
        print(*[str(param) + "=" + str(value) for param, value in vars(args).items()], sep='\n', file=f)

    # Initiate Logger, which records the results on the fly and finally output to the log file.
    # i.e., the mean and std of the metric values over multiple runs.
    if args.eval_metric == "auc":
        loggers = {
            'AUC': Logger(args.runs, args),
        }
        if args.lets_dp:
            loggers["EPS"] = Logger(args.runs, args)
    else:
        raise ValueError(f"Invalid eval_metric {args.eval_metric}!")

    # Creat the path for save processed data.
    # This path is a directory, and the processed files will be saved under this directory.
    # If this path does not exist, a new directory will be created.
    dataset_root = f"./data/processed_data/{args.data_name}"
    dataset_path = dataset_root + '_seal{}'.format(args.data_appendix)

    # Other hyper-params
    max_z = 1000
    emb = None
    val_ratio = 0.1
    directed = False

    # Run the main code
    main()

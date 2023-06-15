import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp

import scipy
from tqdm import tqdm
import pdb

import numpy as np
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

warnings.filterwarnings("ignore", category=UserWarning)

from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *
from GNN.sampler import subsample_graph, subsample_graph_for_undirected_graph

parser = argparse.ArgumentParser(description='SEAL_for_small_dataset')
# Dataset Setting
parser.add_argument('--data_name', type=str, default="NS")

# Subgraph extraction settings
parser.add_argument('--node_label', type=str, default='drnl',
                    help="which specific labeling trick to use")
parser.add_argument('--num_hops', type=int, default=2)
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', default=None)
parser.add_argument('--max_node_degree', default=5)
parser.add_argument('--check_degree_constrained', default=False)
parser.add_argument('--check_degree_distribution', default=True)

# GNN Setting
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)

parser.add_argument('--dynamic_train', action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--eval_metric', default="auc")
parser.add_argument('--hitsK', default=50)

# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers for dynamic mode; 0 if not dynamic")

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


class SEALDatasetSmall(InMemoryDataset):
    def __init__(self, root, A_csc, split_edge, num_hops, node_features=None, percent=100, split='train',
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
            pos_edge_degree_constrained: torch.Tensor = subsample_graph_for_undirected_graph(pos_edge,
                                                                                             max_degree=args.max_node_degree)  # TODO: The train indices should be whole training graph (A_csc)
            neg_edge_degree_constrained: torch.Tensor = subsample_graph_for_undirected_graph(neg_edge,
                                                                                             max_degree=args.max_node_degree)
            # Shuffle and ensure the same number of pos and neg edges
            indexes = np.arange(min(pos_edge_degree_constrained.shape[1], neg_edge_degree_constrained.shape[1]))
            random.shuffle(indexes)
            pos_edge_degree_constrained = pos_edge_degree_constrained[:, indexes]
            neg_edge_degree_constrained = neg_edge_degree_constrained[:, indexes]

            degree_constrained_csc_mat = ssp.csc_matrix((np.ones(len(pos_edge_degree_constrained[0])),
                                                         (pos_edge_degree_constrained[0],
                                                          pos_edge_degree_constrained[1])),
                                                        shape=self.A_csc.shape, dtype=int)
            degree_constrained_csc_mat = degree_constrained_csc_mat + degree_constrained_csc_mat.T - ssp.diags(
                degree_constrained_csc_mat.diagonal(), dtype=int)
            self.A_csc = degree_constrained_csc_mat
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
            assert scipy.linalg.issymmetric(self.A_csc.toarray()), "Train_net must be symmetric!"
            pos_edge = pos_edge_degree_constrained
            neg_edge = neg_edge_degree_constrained
            with open(self.process_log_path, 'a') as f:
                print("After edge sampling", file=f)
                print(f"pos_edge nums:{pos_edge.shape[1]}", file=f)
                print(f"neg_edge nums:{neg_edge.shape[1]}", file=f)

            if args.check_degree_distribution:
                import matplotlib.pyplot as plt
                import pandas as pd
                degree_distribution = np.array(self.A_csc.sum(axis=1)).reshape(-1)
                df = pd.DataFrame(degree_distribution, columns=["degree"])
                # df["binned"] = pd.cut(degree_distribution, bins=10)
                with open(self.process_log_path, 'a') as f:
                    print(f"Degree distribution:", file=f)
                    print(pd.value_counts(df["degree"]), file=f)
                    # print(pd.value_counts(df["binned"]), file=f)
                plt.hist(degree_distribution, bins=len(df["degree"].unique()))
                plt.xlabel("Degree")
                plt.ylabel("Frequency")
                plt.show()

        ####################################
        A_csr = self.A_csc.tocsr()

        if not self.directed:
            self.A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A_csr, self.node_features, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, self.A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A_csr, self.node_features, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, self.A_csc)
        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


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

    return total_loss / len(train_dataset)


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
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    A_csc, split_edge = load_data(args.data_name)

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
    test_dataset = test_dataset.shuffle()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    for run in range(args.runs):
        # model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
        #               train_dataset, args.dynamic_train, use_feature=args.use_feature,
        #               node_embedding=emb).to(device)
        model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, use_feature=args.use_feature,
                    node_embedding=emb).to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')

        # Save model parameters descriptions into log file
        with open(log_file, 'a') as f:
            print(f'Total number of parameters is {total_params}', file=f)

        # Training
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + args.epochs):
            loss = train(model, train_loader, optimizer, train_dataset, emb)
            if epoch % args.eval_steps == 0:
                results = test(model, val_loader, test_loader, emb)

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

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    print(f'Total number of parameters is {total_params}')
    print(f'Results are saved in {args.res_dir}')


if __name__ == "__main__":
    # Create the path for saving data and log
    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")  # Mark the time
    if args.data_appendix == '':  # Create the data save path
        args.data_appendix = '_d{}_h{}'.format(args.max_node_degree, args.num_hops)
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

    # Initiate Logger, which records the results on the fly and finally output to the log file.
    # i.e., the mean and std of the metric values over multiple runs.
    if args.eval_metric == "auc":
        loggers = {
            'AUC': Logger(args.runs, args),
        }
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
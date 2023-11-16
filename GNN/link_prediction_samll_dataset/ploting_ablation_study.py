#!/usr/bin/python
# -*- coding:utf-8 -*-
from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# from utils.Evaluation import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化

plt.rcParams['figure.figsize'] = [10, 8]  # 图片像素
plt.rcParams['savefig.dpi'] = 600  # 分辨率
plt.rcParams["font.family"] = "Times New Roman"


# plt.rcParams['figure.dpi'] = 300 #分辨率


# def Plotings(df_compare, markers, df_name=None, first_column_name="epsilon", metric_name="NDCG@10",
#              x_label=r'${\rm \epsilon}$'):
#     """ A dataframe where the first column is the x label values, and multiple y as multiple lines.
#     Args:
#         df_compare:
#         markers:
#         first_column_name:
#         metric_name:
#         x_label:
#
#     Returns:
#
#     """
#     # plt.figure(figsize=(10,8),dpi=150)
#     fig, ax = plt.subplots()
#     names = df_compare.columns.values
#     for i in range(1, len(df_compare.columns) - 1, 2):
#         # plt.plot(df_compare[first_column_name], df_compare.iloc[:, i], label=names[i],
#         #          marker=markers[i - 1], markersize=20,
#         #          markerfacecolor='none', linestyle='-', color=mcolors.TABLEAU_COLORS[colors[i - 1]], linewidth=3)
#         # ax.plot(df_compare[first_column_name], df_compare.iloc[:, i], label=names[i],
#         #         marker=markers[i - 1], markersize=20,
#         #         linestyle='-', color=mcolors.TABLEAU_COLORS[colors[i - 1]], linewidth=3)
#         plt.errorbar(df_compare[first_column_name], df_compare.iloc[:, i], df_compare.iloc[:, i + 1], label=names[i],
#                      marker=markers[i - 1], markersize=20,
#                      linestyle='-', color=mcolors.TABLEAU_COLORS[colors[i - 1]], linewidth=3)
#     # ax.set_xticks([2, 3, 4, 5])
#     ax.tick_params(axis="x", labelsize=25)
#     ax.tick_params(axis="y", labelsize=25)
#     ax.set_xlim(df_compare.iloc[0, 0], df_compare.iloc[-1, 0])
#     # ax.set_yticks(fontsize=25)
#     ax.set_xlabel(x_label, fontsize=30)
#     ax.set_ylabel(metric_name, fontsize=25)
#     # plt.legend(loc='lower right', prop={'size': 20})
#     ax.grid()
#     plt.savefig(f'./results_figures/{first_column_name}_{df_name}.pdf', format="pdf", dpi=600)
#     plt.show()

def Plotings(df_list: list, markers, labels: list = None, metric_name="NDCG@10",
             x_label=r'${\rm \epsilon}$', save_file_path: str = None, fix: str = None):
    """ A dataframe where the first column is the x label values, and multiple y as multiple lines.
    Args:
        df_compare:
        markers:
        first_column_name:
        metric_name:
        x_label:

    Returns:

    """
    # plt.figure(figsize=(10,8),dpi=150)
    fig, ax = plt.subplots()
    for i, df in enumerate(df_list):
        # names = df.columns.values
        if fix == "num_hop":
            plt.errorbar(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label=labels[i],
                         marker=markers[i], markersize=20,
                         linestyle='-', color=mcolors.TABLEAU_COLORS[colors[i]], linewidth=3,
                         capsize=15, capthick=3)
        elif fix == "max_node_degree":
            plt.bar(df.iloc[:, 0], df.iloc[:, 1], label=labels[i], width=0.5, color=mcolors.TABLEAU_COLORS[colors[i]])
            break
    label_size = 40
    if fix == "num_hop":
        ax.set_ylim(70, 100)
        plt.xticks([20, 40, 60, 80, 100], fontsize=label_size)
        plt.yticks([80, 90, 100], fontsize=label_size)
        ax.set_xlim(df.iloc[0, 0], df.iloc[-1, 0])
        plt.legend(loc='lower right', prop={'size': label_size-5})
    elif fix == "max_node_degree":
        ax.set_ylim(50, 100)
        plt.xticks([2.0, 3.0, 4.0], fontsize=label_size)
        plt.yticks([60, 70, 80, 90, 100], fontsize=label_size)
        ax.set_xlim(df.iloc[0, 0] - 0.5, df.iloc[-1, 0] + 0.5)
    ax.tick_params(axis="x", labelsize=label_size)
    ax.tick_params(axis="y", labelsize=label_size)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(metric_name, fontsize=label_size)
    ax.grid()
    if save_file_path:
        plt.savefig(save_file_path, format="pdf", dpi=600, bbox_inches='tight')
    plt.show()


def load_results(file_path):
    import pickle
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
        # print(f"results:{b}")
    return b


def load_data(compare_res_file_list, ground_truth: float):
    from functools import reduce
    res_df_list = []
    for file_path in compare_res_file_list:
        res_df = pd.read_csv(file_path, index_col=0, header=0)
        temp = res_df[["epsilon", "final_test"]]
        res_df_list.append(temp)
    # comparison_df = pd.merge(res_df_list, on="epsilon", how="inner")
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['epsilon'],
                                                    how='outer'), res_df_list)
    df_merged.columns = ["epsilon", "DPLP-PS", "DPLP-NP"]
    df_merged["GroundTruth"] = ground_truth
    return df_merged


def test_ablation(file_path, compare_target, x_label):
    # The results from multiple datasets.
    res_df = pd.read_csv(file_path, index_col=0, header=0)
    data_degree_groups = res_df.groupby(['dataset'])
    temp = []
    for name, group in data_degree_groups:
        group = group.sort_values(compare_target, axis=0)
        sub_df = group[[compare_target, "final_test"]]
        Plotings(sub_df, compares_markers, metric_name="AUC(%)", x_label=x_label)


def load_table(file_path, type="csv", dtypes: dict = None):
    if type == "csv":
        dtypes = object if not dtypes else dtypes
        res_df = pd.read_csv(file_path, index_col=0, header=0, dtype=dtypes)
    else:
        raise ValueError("invalid data type")
    return res_df


def data_extraction(query: str):
    from pandasql import sqldf
    pysqldf = lambda q: sqldf(q, globals())
    pandasSQL_solution = pysqldf(query)
    return pandasSQL_solution


def query_generate_hop(**kwargs):
    y, std = "final_test", "test_std"
    x = kwargs.get("x")
    epsilon = kwargs.get("epsilon")
    max_node_degree = kwargs.get("max_node_degree")
    data_name = kwargs.get("data_name")
    num_hop = kwargs.get("num_hop")
    query = f"""
        SELECT {x}, {y}, {std} FROM df
        WHERE df.dataset == '{data_name}' AND df.epsilon == {epsilon} AND df.max_node_degree == {max_node_degree}
        ORDER BY df.num_hop ASC
        """
    save_file_path = f'./results_figures/{data_name}_{x}_{max_node_degree}.pdf'
    return query, save_file_path


def query_generate(**kwargs):
    y, std = "final_test", "test_std"
    x = kwargs.get("x")
    epsilon = kwargs.get("epsilon")
    max_node_degree = kwargs.get("max_node_degree")
    data_name = kwargs.get("data_name")
    num_hop = kwargs.get("num_hop")
    query = f"""
            SELECT {x}, {y}, {std} FROM df 
            WHERE df.dataset == '{data_name}' AND df.epsilon == {epsilon} AND df.num_hop == {num_hop}
            ORDER BY df.num_hop ASC
            """
    save_file_path = f'./results_figures/{data_name}_{x}_{max_node_degree}_{num_hop}.pdf'
    return query, save_file_path


def main(fix="max_node_degree"):
    queries, df_list = [], []
    data_name = "Celegans"
    if fix == "max_node_degree":
        # OPTION: FIX NODE DEGREE AND CHECK HOP
        max_node_degree = 60
        key_values = [{"x": "num_hop", "epsilon": 11, "data_name": f"{data_name}", "max_node_degree": max_node_degree},
                      {"x": "num_hop", "epsilon": 3, "data_name": f"{data_name}", "max_node_degree": max_node_degree}]
    elif fix == "num_hop":
        num_hop = 2
        key_values = [{"x": "max_node_degree", "epsilon": 11, "data_name": f"{data_name}", "num_hop": num_hop},
                      {"x": "max_node_degree", "epsilon": 3, "data_name": f"{data_name}", "num_hop": num_hop}]
    else:
        raise ValueError("not a valid fix")
    for ky in key_values:
        if fix == "max_node_degree":
            query, save_file_path = query_generate_hop(**ky)
        elif fix == "num_hop":
            query, save_file_path = query_generate(**ky)
        else:
            raise ValueError("not a valid fix")
        queries.append(query)

    for query in queries:
        df = data_extraction(query)
        df_list.append(df)
    # save_file_path = f'./results_figures/{"Celegans"}_{"num_hop"}.pdf'
    if fix == "max_node_degree":
        x_label = "k"
    elif fix == "num_hop":
        x_label = r"$\theta$"
    Plotings(df_list, compares_markers, labels=[r"$\varepsilon$=11", r"$\varepsilon$=3"], metric_name="AUC(%)", x_label=x_label,
             save_file_path=save_file_path, fix=fix)


if __name__ == '__main__':
    # compare_res_file_list = ["./results/all_results_20230810192501.csv",
    #                          "./results/all_results_20230810195856.csv"] # Yeast
    # compare_res_file_list = ["./results/all_results_20230810192501.csv",
    #                          "./results/all_results_20230810204144.csv"]  # Yeast
    # compare_res_file_list = ["./results/all_results_20230810223909.csv",
    #                          "./results/all_results_20230810225457.csv"]  # NS
    # compare_res_file_list = ["./results/all_results_20230811000952.csv",
    #                          "./results/all_results_20230811001429.csv"]  # USAir
    compare_res_file_list = ["./results/all_results_20230811012141.csv",
                             "./results/all_results_20230811011100.csv"]  # PB
    # file_path = "./results/ablation_studies_hop.csv"
    # file_path = "./results/ablation_studies_max_node_degree.csv"
    # file_path = "./results/ablation_studies_num_hops_20230904.csv"
    file_path = "./results/all_results_2023-08-27-03-04-58.csv"
    # df_compare = load_data(compare_res_file_list, 93.)
    # compares = [compare1, compare2, compare3, compare4, compare5]
    # compares = [compare1, compare2]
    # compares_names = ['k=5', 'k=7', 'k=10', 'k=20']
    # compares_names = ['IDPNMF', 'DPNMF', 'DPMF', 'DPSGD', 'ALSOPP']
    compares_markers = ['s', 'o', '^', '*', 'v', 'x', 'D']
    # compares_colors = ['red', 'green', 'blue', 'purple', 'black', 'brown']
    # Plotings(compares_names, compares_colors, compares_markers, *compares)
    # Plotings(df_compare, compares_markers, metric_name="AUC(%)")
    # test_ablation(file_path, "num_hop", x_label=r'${\theta}$')
    dtypes = {"max_node_degree": int, "epsilon": int, "dataset": str}
    df = load_table(file_path, dtypes=dtypes)
    main(fix="num_hop")


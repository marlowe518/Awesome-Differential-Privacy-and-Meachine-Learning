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


def Plotings(df_compare, markers, df_name=None, first_column_name="epsilon", metric_name="NDCG@10",
             x_label=r'${\rm \epsilon}$'):
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
    names = df_compare.columns.values
    for i in range(1, len(df_compare.columns)):
        # plt.plot(df_compare[first_column_name], df_compare.iloc[:, i], label=names[i],
        #          marker=markers[i - 1], markersize=20,
        #          markerfacecolor='none', linestyle='-', color=mcolors.TABLEAU_COLORS[colors[i - 1]], linewidth=3)
        ax.plot(df_compare[first_column_name], df_compare.iloc[:, i], label=names[i],
                marker=markers[i - 1], markersize=20,
                linestyle='-', color=mcolors.TABLEAU_COLORS[colors[i - 1]], linewidth=3)
    # ax.set_xticks([2, 3, 4, 5])
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    ax.set_xlim(df_compare.iloc[0, 0], df_compare.iloc[-1, 0])
    # ax.set_yticks(fontsize=25)
    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(metric_name, fontsize=25)
    # plt.legend(loc='lower right', prop={'size': 20})
    ax.grid()
    plt.savefig(f'./results_figures/{first_column_name}_{df_name}.pdf', format="pdf", dpi=600)
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
        Plotings(sub_df, compares_markers, name, compare_target, metric_name="AUC(%)", x_label=x_label)


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
    file_path = "./results/ablation_studies_max_node_degree.csv"
    # df_compare = load_data(compare_res_file_list, 93.)
    # compares = [compare1, compare2, compare3, compare4, compare5]
    # compares = [compare1, compare2]
    # compares_names = ['k=5', 'k=7', 'k=10', 'k=20']
    # compares_names = ['IDPNMF', 'DPNMF', 'DPMF', 'DPSGD', 'ALSOPP']
    compares_markers = ['s', 'o', '^', '*', 'v', 'x', 'D']
    # compares_colors = ['red', 'green', 'blue', 'purple', 'black', 'brown']
    # Plotings(compares_names, compares_colors, compares_markers, *compares)
    # Plotings(df_compare, compares_markers, metric_name="AUC(%)")
    test_ablation(file_path, "max_node_degree", x_label=r'${\theta}$')

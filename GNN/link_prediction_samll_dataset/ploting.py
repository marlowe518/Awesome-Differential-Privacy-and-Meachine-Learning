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
plt.rcParams['savefig.dpi'] = 1000  # 分辨率


# plt.rcParams['figure.dpi'] = 300 #分辨率


def Plotings(df_compare, markers, metric_name="NDCG@10", x_label=r'${\rm \epsilon}$'):
    fig = plt.figure()
    names = df_compare.columns.values
    for i in [1, 2, 3]:
        plt.plot(df_compare['epsilon'], df_compare.iloc[:, i], label=names[i],
                 marker=markers[i - 1], markersize=20,
                 markerfacecolor='none', linestyle='--', color=mcolors.TABLEAU_COLORS[colors[i - 1]], linewidth=3)
    plt.xticks(fontsize=25)
    plt.xlim(df_compare['epsilon'][0], df_compare['epsilon'][len(df_compare) - 1])
    plt.yticks(fontsize=25)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(metric_name, fontsize=23)
    plt.legend(loc='lower right', prop={'size': 20})
    plt.grid()
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
    df_compare = load_data(compare_res_file_list, 93.)
    # compares = [compare1, compare2, compare3, compare4, compare5]
    # compares = [compare1, compare2]
    # compares_names = ['k=5', 'k=7', 'k=10', 'k=20']
    # compares_names = ['IDPNMF', 'DPNMF', 'DPMF', 'DPSGD', 'ALSOPP']
    compares_markers = ['o', '^', '*', 'v', 'x', 'D']
    # compares_colors = ['red', 'green', 'blue', 'purple', 'black', 'brown']
    # Plotings(compares_names, compares_colors, compares_markers, *compares)
    Plotings(df_compare, compares_markers, metric_name="AUC(%)")

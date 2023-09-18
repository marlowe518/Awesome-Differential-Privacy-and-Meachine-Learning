import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd


def save_results(file_path, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(file_path):
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
        # print(f"results:{b}")
    return b


if __name__ == "__main__":
    all_dicts = []
    # begin_date = ["20230805","20230806"]
    coarse_filter = ["2023-09-17", "2023-09-17"]
    begin_time_str = "2023-09-17-19-06-00"
    end_time_str = "2023-09-18-03-41-00"
    # end_time_str = "2023-12-31-23-59-59"
    begin_date = datetime.datetime.strptime(begin_time_str, "%Y-%m-%d-%H-%M-%S")
    end_date = datetime.datetime.strptime(end_time_str, "%Y-%m-%d-%H-%M-%S")
    # begin_date = ['Ecoli_20230811']
    for res_file_dir in os.listdir('./results'):
        date = res_file_dir.split("_")[-1].split(".")[0]  # date time is the last item after "_"
        if any([d in res_file_dir for d in coarse_filter]):
            date_time_comparable = datetime.datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
            if (date_time_comparable > begin_date) and (date_time_comparable <= end_date):
                res_file = f"./results/{res_file_dir}/key_results.pickle"
                if os.path.isfile(res_file):
                    all_dicts.append(load_results(res_file))
        # OPTION 2:
        # for date in begin_date:
        #     if date in res_file_dir:  # TODO 改为大于该日期
        #         res_file = f"./results/{res_file_dir}/key_results.pickle"
        #         if os.path.isfile(res_file):
        #             all_dicts.append(load_results(res_file))

    from collections import defaultdict

    # combined_res = dict.fromkeys()
    input_keys = ['dataset', 'experiment', 'max_node_degree', 'num_hop', 'highest_val', 'final_test', 'val_std',
                  'test_std', 'best_epoch', 'final_round_val', 'final_round_val_std', 'final_round_test',
                  'final_round_test_std', 'original_edges', 'sampled_edges', 'max_term_per_edge', 'epsilon', 'sigma',
                  'all_runs', 'dp_method', 'sens', 'parameter_indicator', 'eps', 'lr', 'max_norm', 'batch_size',
                  'train_samples', 'neighborhood_subgraph']
    combined_res = dict([(i, []) for i in input_keys])
    # combined_res = defaultdict(list)
    for d in all_dicts:  # you can list as many input dicts as you want here
        if (["sens", "parameter_indicator", "eps"] not in d.keys()).all():
            d["sens"] = ['n/a']
            d["parameter_indicator"] = ['n/a']
            d["eps"] = ['n/a']
        for key, value in d.items():
            combined_res[key].append(value)
    res_df = pd.DataFrame.from_dict(combined_res)
    # res_df = res_df.fillna(method='ffill')
    # res_df = res_df.fillna(value=-1)
    data_degree_groups = res_df.groupby(['dataset', 'max_node_degree'])
    temp = []
    for names, group in data_degree_groups:
        group["original_edges"] = group["original_edges"].fillna(method='ffill')
        group["sampled_edges"] = group["sampled_edges"].fillna(method='ffill')
        temp.append(group)
    results = pd.concat(temp)
    res_df = results
    # results = data_degree_groups.apply(lambda x:x)
    # res_df["original_edges"] = res_df["original_edges"].fillna(method='ffill')
    # res_df["sampled_edges"] = res_df["sampled_edges"].fillna(method='ffill')
    res_df["val_trend"] = res_df["all_runs"].apply(lambda x: ",".join(
        list(np.ravel(np.mean(np.array(x["val_test_trend"]), axis=0)[:, 0]).astype(str))))  # save as ravelled string
    # print(res_df["val_test_trend"][0], sep="\n")
    res_df = res_df.drop(["all_runs"], axis=1)
    res_df.to_csv(f"./results/all_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
    print(res_df)

    ##### temp #######
    # import numpy as np
    #
    # temp = np.hstack([np.array(res_df["val_test_trend"].values[0]), np.array(res_df["eps_trend"].values[0])])
    # first_exceed_index = []
    # for i in range(1, 11, 1):
    #     try:
    #         index_arr = np.where(temp[:, 2] > i)
    #         if len(index_arr[0]) == 0:
    #             continue
    #         index = index_arr[0][0]
    #         first_exceed_index.append((i, temp[:, 2][index], temp[:, 1][index]))
    #     except Exception as e:
    #         print("error")
    # print(f"epsilon trend: \n{temp}")
    # print(first_exceed_index)
    # import matplotlib.pyplot as plt
    # plt.plot(temp[:,2][:int(res_df["best_epoch"])+5], temp[:,1][:int(res_df["best_epoch"])+5])
    # plt.show()

    # inspect
    # import matplotlib.pyplot as plt
    #
    # ratio = (res_df["sampled_edges"] / res_df["max_term_per_edge"]) / 200
    # plt.plot(res_df["final_test"], label="final_test")
    # plt.plot(ratio, label="indicator")
    # # plt.plot(res_df["parameter_indicator"], label="indicator")
    # plt.legend(loc="best")
    # plt.show()

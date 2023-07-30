import os
import pickle
import time

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
    begin_date = ["20230714"]
    begin_date = ["NS_20230730234908"]
    for res_file_dir in os.listdir('./results'):
        for date in begin_date:
            if date in res_file_dir:  # TODO 改为大于该日期
                res_file = f"./results/{res_file_dir}/key_results.pickle"
                if os.path.isfile(res_file):
                    all_dicts.append(load_results(res_file))

    from collections import defaultdict

    combined_res = defaultdict(list)
    for d in all_dicts:  # you can list as many input dicts as you want here
        for key, value in d.items():
            combined_res[key].append(value)
    res_df = pd.DataFrame.from_dict(combined_res)
    res_df = res_df.fillna(method='ffill')
    res_df.to_csv(f"./results/all_results_{time.strftime('%Y%m%d%H%M%S')}.csv")
    print(res_df)

    ##### temp #######
    import numpy as np
    temp = np.hstack([np.array(res_df["val_test_trend"].values[0]), np.array(res_df["eps_trend"].values[0])])
    print(f"epsilon trend: \n{temp}")
    # print(res_df["val_test_trend"].values)
    # print(res_df["eps_trend"].values)

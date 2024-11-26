#!/usr/bin/python
# -*- coding:utf-8 -*-

# @Time : 2022/2/10 11:34 上午
# @Author : rx
# @File : sample.py
# @Software: PyCharm
# @contact: 3215907348@qq.com

# -*- 功能说明 -*-

# 负责对训练数据进行采样

# -*- 功能说明 -*-

import pandas as pd
import numpy as np


def sampling(data: pd.DataFrame, privacy_spec, user_priv_pref, threshold, random_seed):
    '''
    :param data: DataFrame
    :param privacy_spec: dict element wise privacy level
    :param user_priv_pref: dict user level privacy preferences
    :param threshold: sampling threshold (no need in avg type)
    :param random_seed: seed for random var sampling
    :return: dataframe sampled data
    '''
    np.random.seed(random_seed)
    sampled_indexes = []

    # get the threshold t # TODO 这里可以优化一下，应该不需要两个循环来做。
    all_eps = []
    for index, row in data.iterrows():
        u = str(int(row['uid']))
        i = str(int(row['iid']))
        if (u not in privacy_spec) or (i not in privacy_spec[u]):
            eps_ui = user_priv_pref[u]
        else:
            eps_ui = privacy_spec[u][i]
        all_eps.append(eps_ui)
    t = np.average(all_eps)

    # sampling
    for index, row in data.iterrows():
        u = str(int(row['uid']))
        i = str(int(row['iid']))
        if (u not in privacy_spec) or (i not in privacy_spec[u]):
            eps_ui = user_priv_pref[u]
        else:
            eps_ui = privacy_spec[u][i]

        if t <= eps_ui:  # if the threshold is less or equal to eps_ui, the rui is selected
            sampled_indexes.append(index)
            continue
        else:
            prob = (np.exp(eps_ui) - 1) / (np.exp(t) - 1)  # compute the selection probability
            select = np.random.choice([1, 0], p=[prob, 1 - prob])  # select according to the prob
            if select:
                sampled_indexes.append(index)  # if selected, the corresponding index is added into the sampled array
    sampled_data = data.loc[sampled_indexes]  # get the sampled data according to the indexes
    print(f"sample out nums: {len(data) - len(sampled_data)}")
    return sampled_data, t

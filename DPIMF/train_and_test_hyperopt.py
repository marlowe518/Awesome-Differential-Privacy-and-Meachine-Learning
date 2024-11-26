#!/usr/bin/python
# -*- coding:utf-8 -*-

# @Time : 2022/2/17 10:49 上午
# @Author : rx
# @File : train_and_test_hyperopt.py
# @Software: PyCharm
# @contact: 3215907348@qq.com

# -*- 功能说明 -*-

# 用于超参数调优

# -*- 功能说明 -*-
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from train_and_test import main as dpoccf
from absl import app, flags
import sys
import pandas as pd
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "data path")
flags.DEFINE_integer("evals", None, "times of the hyper parameter search")


# flags.DEFINE_integer("evals", None, "times of the hyper parameter search")
# flags.DEFINE_string("orig_model", None, "orginal model")
# flags.DEFINE_string("spec_model", None, "specific model")
# flags.DEFINE_boolean("normalization", None, "data normalization")
# flags.DEFINE_string("dataset", None, "dataset used to train and test")


# 将控制台输出保存到文件
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def objective(params: dict):
    # epochs = params.get("epochs")
    embedding_dim = params.get('embedding_dim')
    reg = params.get('reg')
    # unobserved_weight = params.get('unob_weight')

    result = dpoccf(data=FLAGS.data,
                    epochs=10,
                    embedding_dim=embedding_dim,
                    regularization=reg,
                    unobserved_weight=1.0,
                    stddev=0.1,
                    epsilon=10,
                    privacy_specification=None,
                    user_privacy_pref=None,
                    random_seed=1)
    return {'loss': result, 'status': STATUS_OK}


def main(argv):
    sys.stdout = Logger(filename=f"ml-100k.log",
                        stream=sys.stdout)
    reg_list = [0.0, 0.001, 0.01, 0.1, 1.0]
    # reg_list = [0.05, 0.07, 0.1, 0.3, 0.7]
    emb_dim_list = [8, 16, 24, 48, 64]
    # emb_dim_list = [16]
    # emb_dim_list = [5, 6, 7, 8]
    # unob_weight_list = [0.01, 0.1, 0.5, 1.0]
    # unob_weight_list = [1.0]

    space = {
        "reg": hp.choice("reg", reg_list),
        "embedding_dim": hp.choice("embedding_dim", emb_dim_list),
        # "unob_weight": hp.choice("unob_weight", unob_weight_list)
    }
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=FLAGS.evals, trials=trials)
    best_params = [(param, best[param]) for param in space.keys()]
    print("best params:")
    print(*best_params, sep='\n')
    print(*trials, sep='\n')


def grid_search(argv):
    reg_list = [0.0, 0.001, 0.01, 0.1, 1.0]
    # reg_list = [0.1, 1.0, 3.0, 5.0, 10]
    emb_dim_list = [2, 8, 16, 24, 48, 64]
    # reg_list = [0.001]
    # emb_dim_list = [2, 4, 6, 8]
    # reg_list = [0.1, 0.2]
    # emb_dim_list = [5]
    best_result = 9999
    best_params = [0, 0]
    result_list = []
    repeats = 1
    for reg in reg_list:
        for emb_dim in emb_dim_list:
            print(f"params:{reg, emb_dim}")
            record = []
            for repeat in range(repeats):
                print(f'repeat:{repeat}')
                result = dpoccf(data=FLAGS.data,
                                epochs=10,
                                embedding_dim=emb_dim,
                                regularization=reg,
                                unobserved_weight=1.0,
                                stddev=0.1,
                                epsilon=10,
                                privacy_specification=None,
                                user_privacy_pref=None,
                                random_seed=1)
                record.append(result)
            result = np.mean(record)
            if result < best_result:
                best_result = result
                best_params = [reg, emb_dim]
            print(f"best results: {best_result}")
            print(f"best parmas: {best_params}")
            result_list.append([reg, emb_dim, result])
    pd.DataFrame(result_list).to_csv(FLAGS.data + '_grid_search.csv', sep=',')
    return result_list


if __name__ == '__main__':
    # app.run(main)
    # datapath = 'data/yahoomusic/yahoo_music'
    # grid_search('data/yahoomusic/yahoo_music')
    # grid_search('data/IV/amazon')
    # grid_search('data/ml-100k/ml-100k')\
    app.run(grid_search)

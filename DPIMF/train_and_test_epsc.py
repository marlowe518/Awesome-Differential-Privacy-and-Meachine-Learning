#!/usr/bin/python
# -*- coding:utf-8 -*-

# @Time : 2022/2/12 3:42 下午
# @Author : rx
# @File : train_and_test_epsc.py
# @Software: PyCharm
# @contact: 3215907348@qq.com

# -*- 功能说明 -*-

#

# -*- 功能说明 -*-
import argparse
import numpy as np
import pandas as pd
import argparse
from utils.LoadData import DataSet
from ials import DataSet as IALSDataSet
from ials import IALS
# from dpoccf_gramian import IALS as DPIALS
from dpoccf import IALS as DPIALS
from evaluate import evaluate_model
from time import time
import pickle
from get_privacy_spec import getUserPrivacyLevel, urpToudp, load_dict
from risk_pattern_mining import loadData
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "data path")
flags.DEFINE_string("train_data", None, "train data path")
flags.DEFINE_string("urp_path", None, "urp path")
flags.DEFINE_string("sens_type", None, "type of sensitivity")
flags.DEFINE_float("reg", None, "reg")
flags.DEFINE_float("unob", None, "unobserved weight")
flags.DEFINE_integer("d", None, "dimension")
flags.DEFINE_integer("evals", None, "times of the hyper parameter search")
flags.DEFINE_integer("seed", None, "random seed")


class MFModel(DPIALS):

    def _predict_one(self, user, item):
        """Predicts the score of a user for an item."""
        if user >= self.user_embedding.shape[0] or item >= self.item_embedding.shape[0]:
            print(f"(u,i):{(user, item)} out of bounds {(self.user_embedding.shape[0], self.item_embedding.shape[0])}")
            return -9999  # no recommending on unpredictable point
        try:
            np.dot(self.user_embedding[user],
                   self.item_embedding[item])
        except:
            print('error')

        return np.dot(self.user_embedding[user],
                      self.item_embedding[item])

    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])  # A list fulfilling user ids
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        for i in range(num_examples):
            predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
        return predictions


def evaluate(model, test_ratings, test_negatives, K=10):
    """Helper that calls evaluate from the NCF libraries."""
    (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, K=K,
                                   num_thread=1)
    return np.array(hits).mean(), np.array(ndcgs).mean()


def main(argv):
    # Command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default='data/ml-100k/ml-100k',
    #                     help='Path to the train dataset')
    # parser.add_argument('--train_data', type=str, default='data/ml-100k/ml-100k_train_data',
    #                     help='Path to the train dataset')
    # # parser.add_argument('--data', type=str, default='data/yahoomusic/yahoo_music',
    # #                     help='Path to the train dataset')
    # # parser.add_argument('--train_data', type=str, default='data/yahoomusic/yahoo_music_train_data',
    # #                     help='Path to the train dataset')
    # # parser.add_argument('--data', type=str, default='data/IV/amazon',
    # #                     help='Path to the train dataset')
    # # parser.add_argument('--train_data', type=str, default='data/IV/amazon_train_data',
    # #                     help='Path to the train dataset')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of training epochs')
    # parser.add_argument('--embedding_dim', type=int, default=6,
    #                     help='Embedding dimensions, the first dimension will be'
    #                          'used for the bias.')
    # parser.add_argument('--regularization', type=float, default=0.07,
    #                     help='L2 regularization for user and item embeddings.')
    # parser.add_argument('--unobserved_weight', type=float, default=1.0,
    #                     help='weight for unobserved pairs.')
    # parser.add_argument('--stddev', type=float, default=0.1,
    #                     help='Standard deviation for initialization.')
    # parser.add_argument('--epsilon', type=float, default=10,
    #                     help='Privacy Budget')
    # parser.add_argument('--urp_path', type=str, default='data/ml-100k/urp_m_2_p_2')
    # # parser.add_argument('--urp_path', type=str, default='data/yahoomusic/urp_m_2_p_2')
    # # parser.add_argument('--urp_path', type=str, default='data/IV/urp_m_2_p_2')
    # # parser.add_argument('--privacy_specification', type=str, default='data/ml-100k/upp')
    # # parser.add_argument('--user_privacy_pref', type=str, default='data/ml-100k/uppp')
    # parser.add_argument('--random_seed', type=int, default=2)
    # args = parser.parse_args()

    # load the configurations
    # data = None
    epochs = 10
    # embedding_dim = None
    # regularization = None
    # unobserved_weight = None
    stddev = 0.1
    epsilon = None
    privacy_specification = None
    user_privacy_pref = None
    random_seed = FLAGS.seed
    repeats = 4
    # sens_type = '1'
    embedding_dim = FLAGS.d
    regularization = FLAGS.reg
    data = FLAGS.data
    train_data = FLAGS.train_data
    unobserved_weight = FLAGS.unob
    sens_type = FLAGS.sens_type
    urp_path = FLAGS.urp_path
    final_results = []
    eps_m = 0.5
    fc = 0.54
    fl = 0.09

    # embedding_dim = 8
    # regularization = 0.01
    # data = 'data/yahoomusic/yahoo_music'
    # unobserved_weight = 0.8
    # sens_type = '1'

    # Generate user privacy preferences
    Data = loadData(train_data)
    for eps_c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # for eps_c in [0.1, 0.2]:
        record = []
        for repeat in range(repeats):
            user_ids = Data.keys()
            user_priv_pref = getUserPrivacyLevel(len(user_ids), eps_c=eps_c, eps_m=eps_m, eps_l=1.0,
                                                 prob_c=fc, prob_m=1. - fc - fl, prob_l=fl,
                                                 random_seed=None)
            user_priv_pref = dict(zip(user_ids, user_priv_pref))

            # Load urp and produce udp and upp
            urp = load_dict(urp_path)  # TODO Does the keys cover all users?  Are there users have no risk pattern?
            _, privacy_spec = urpToudp(urp, user_priv_pref)

            # Load the item-rating data, and user-rating data, testing data and neg testing data
            ds = DataSet(data, privacy_spec, user_priv_pref, random_seed)
            print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' % (
                ds.user_nums, ds.item_nums, ds.record_nums, len(ds.test_ratings)))

            # Prepare data for training
            train_ds = IALSDataSet(ds.train_by_users, ds.train_by_items, ds.test_ratings, 1)

            # Initialize the model
            model = MFModel(ds.user_nums, ds.item_nums,
                            embedding_dim, regularization,
                            unobserved_weight,
                            stddev / np.sqrt(embedding_dim),
                            epsilon)

            # Train and evaluate model
            hr, ndcg = evaluate(model, ds.test_ratings, ds.test_neg_ratings, K=10)
            print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
                  % (0, hr, ndcg))
            for epoch in range(epochs):
                # Training
                start_time = time()
                _ = model.train(train_ds)
                end_time = time()

                # Evaluation
                hr, ndcg = evaluate(model, ds.test_ratings, ds.test_neg_ratings, K=10)
                # print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
                #       % (epoch + 1, hr, ndcg))
                # print("Training time:{}".format(end_time - start_time))

            start_time = time()
            _ = model.train_private(train_ds, ds.threshold)  # update the item embedding
            end_time = time()
            # Evaluation
            hr, ndcg = evaluate(model, ds.test_ratings, ds.test_neg_ratings, K=10)
            # print("Training time:{}".format(end_time - start_time))
            record.append([ds.threshold, hr, ndcg])
        avg_record = np.mean(record, axis=0)
        final_results.append(avg_record)
        t, hr, ndcg = avg_record
        print('EPS=%.2f, HR=%.4f, NDCG=%.4f\t' % (t, hr, ndcg))
    pd.DataFrame(final_results).to_csv(data + '_epsc_rusults.csv', index=False, header=False)


if __name__ == '__main__':
    # # for eps_m in [0.15, 0.2, 0.3, 0.4, 0.55]:
    # for eps_c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     # for eps_c in [0.2]:
    #     # for eps_m in [0.55]:
    #     print(f"eps_c:{eps_c}")
    #     main(eps_c=eps_c)
    app.run(main)

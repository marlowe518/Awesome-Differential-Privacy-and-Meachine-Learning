# Copyright 2021 Ranxun.
import numpy as np
import pandas as pd
import argparse
from utils.LoadData import DataSet
from ials import DataSet as IALSDataSet
from ials import IALS
from dpoccf import IALS as DPIALS
from dpoccf_gramian import IALS as GDPIALS
from evaluate import evaluate_model
from absl import app, flags
from time import time
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "data path")
flags.DEFINE_string("sens_type", None, "type of sensitivity")
flags.DEFINE_float("reg", None, "reg")
flags.DEFINE_float("unob", None, "unobserved weight")
flags.DEFINE_integer("d", None, "dimension")
flags.DEFINE_integer("evals", None, "times of the hyper parameter search")
flags.DEFINE_integer("seed", None, "random seed")

class GMFModel(GDPIALS):

    def _predict_one(self, user, item):
        """Predicts the score of a user for an item."""
        if user >= self.user_embedding.shape[0] or item >= self.item_embedding.shape[0]:
            # print(f"(u,i):{(user, item)} out of bounds {(self.user_embedding.shape[0], self.item_embedding.shape[0])}")
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


class MFModel(DPIALS):

    def _predict_one(self, user, item):
        """Predicts the score of a user for an item."""
        if user >= self.user_embedding.shape[0] or item >= self.item_embedding.shape[0]:
            # print(f"(u,i):{(user, item)} out of bounds {(self.user_embedding.shape[0], self.item_embedding.shape[0])}")
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


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main(argv):
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
    unobserved_weight = FLAGS.unob
    sens_type = FLAGS.sens_type
    # embedding_dim = 8
    # regularization = 0.01
    # data = 'data/yahoomusic/yahoo_music'
    # unobserved_weight = 0.8
    # sens_type = '1'

    # Load the item-rating data, and user-rating data, testing data and neg testing data
    if (privacy_specification is not None) and (user_privacy_pref is not None):
        privacy_spec = load_dict(privacy_specification)
        user_priv_pref = load_dict(user_privacy_pref)
    else:
        privacy_spec, user_priv_pref = None, None
    ds = DataSet(data, privacy_spec, user_priv_pref, random_seed)
    print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' % (
        ds.user_nums, ds.item_nums, ds.record_nums, len(ds.test_ratings)))

    # Prepare data for training
    train_ds = IALSDataSet(ds.train_by_users, ds.train_by_items, ds.test_ratings, 1)

    # Initialize the model
    if sens_type == '1':
        model = GMFModel(ds.user_nums, ds.item_nums,
                         embedding_dim, regularization,
                         unobserved_weight,
                         stddev / np.sqrt(embedding_dim),
                         epsilon)
    else:
        model = MFModel(ds.user_nums, ds.item_nums,
                        embedding_dim, regularization,
                        unobserved_weight,
                        stddev / np.sqrt(embedding_dim),
                        epsilon, sens_type=sens_type)

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
        print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
              % (epoch + 1, hr, ndcg))
        print("Training time:{}".format(end_time - start_time))

    # Obtain user embedding
    # Compute the sensitivity based on user embedding
    # Train the private item embedding
    # Training DPOCCF
    final_results = []
    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # for epsilon in [1.0]:
        # for K in range(2, 11):
        K = 10
        record = []
        for repeat in range(repeats):
            print(f'repeat:{repeat}')
            start_time = time()
            _ = model.train_private(train_ds, epsilon)  # update the item embedding
            end_time = time()
            # Evaluation
            hr, ndcg = evaluate(model, ds.test_ratings, ds.test_neg_ratings, K=K)
            # print('EPS=%.2f, K=%d, HR=%.4f, NDCG=%.4f\t' % (epsilon, K, hr, ndcg))
            # print('K=%d, HR=%.4f, NDCG=%.4f\t' % (K, hr, ndcg))
            # print("Training time:{}".format(end_time - start_time))
            record.append([hr, ndcg])
        avg_record = np.mean(record, axis=0)
        final_results.append(avg_record)
        hr, ndcg = avg_record
        print('EPS=%.2f, K=%d, HR=%.4f, NDCG=%.4f\t' % (epsilon, K, hr, ndcg))
        # print('K=%d, HR=%.4f, NDCG=%.4f\t' % (K, hr, ndcg))
        # print("Training time:{}".format(end_time - start_time))
    pd.DataFrame(final_results).to_csv(data + str(sens_type) + '_sens_rusults.csv', index=False, header=False)
    return -hr


if __name__ == '__main__':
    # udp_file = 'data/ml-100k/udp'
    # upp_file = 'data/ml-100k/upp'
    # upp = load_dict(upp_file)
    # udp = load_dict(udp_file)
    # main(data='data/yahoomusic/yahoo_music',
    #      epochs=10,
    #      embedding_dim=8,
    #      regularization=0.01,
    #      unobserved_weight=0.8,
    #      stddev=0.1,
    #      epsilon=10,
    #      privacy_specification=None,
    #      user_privacy_pref=None,
    #      random_seed=4,
    #      sens_type='1')
    app.run(main)

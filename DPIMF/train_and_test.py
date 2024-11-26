# Copyright 2021 Ranxun.
import numpy as np
import argparse
from utils.LoadData import DataSet
from ials import DataSet as IALSDataSet
from ials import IALS
from dpoccf import IALS as DPIALS
from evaluate import evaluate_model
from time import time
import pickle


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


def main(data,
         epochs,
         embedding_dim,
         regularization,
         unobserved_weight,
         stddev,
         epsilon,
         privacy_specification=None,
         user_privacy_pref=None,
         random_seed=1,
         clip=False):
    # Command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default='data/ml-100k/ml-100k',
    #                     help='Path to the dataset')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of training epochs')
    # parser.add_argument('--embedding_dim', type=int, default=8,
    #                     help='Embedding dimensions, the first dimension will be'
    #                          'used for the bias.')
    # parser.add_argument('--regularization', type=float, default=0.0,
    #                     help='L2 regularization for user and item embeddings.')
    # parser.add_argument('--unobserved_weight', type=float, default=1.0,
    #                     help='weight for unobserved pairs.')
    # parser.add_argument('--stddev', type=float, default=0.1,
    #                     help='Standard deviation for initialization.')
    # parser.add_argument('--epsilon', type=float, default=10,
    #                     help='Privacy Budget')
    # parser.add_argument('--privacy_specification', type=str, default='data/ml-100k/upp')
    # parser.add_argument('--user_privacy_pref', type=str, default='data/ml-100k/uppp')
    # parser.add_argument('--random_seed', type=int, default=1)
    # args = parser.parse_args()
    # args.privacy_specification, args.user_privacy_pref = None, None  # Setting for DPOCCF

    # load the raw data

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
    model = MFModel(ds.user_nums, ds.item_nums,
                    embedding_dim, regularization,
                    unobserved_weight,
                    stddev / np.sqrt(embedding_dim),
                    epsilon,
                    clipping=clip)

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
    # for epsilon in np.arange(0.1, 1.1, 0.1):
    # for epsilon in [0.1, 1, 5, 10, 20]:
    # for epsilon in [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]:
    # for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
    # for epsilon in [0.5]:
    # for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for epsilon in [1.0]:
        start_time = time()
        _ = model.train_private(train_ds, epsilon)  # update the item embedding
        end_time = time()
        # Evaluation
        hr, ndcg = evaluate(model, ds.test_ratings, ds.test_neg_ratings, K=10)
        print(f'Sensitivity: {model.sens}')
        print('EPS=%.2f, HR=%.4f, NDCG=%.4f\t' % (epsilon, hr, ndcg))
        print("Training time:{}".format(end_time - start_time))

    # Training RPDPOCCF
    # start_time = time()
    # _ = model.train_private(train_ds, ds.threshold)  # update the item embedding
    # end_time = time()
    # # Evaluation
    # hr, ndcg = evaluate(model, ds.test_ratings, ds.test_neg_ratings, K=10)
    # print('EPS=%.2f, HR=%.4f, NDCG=%.4f\t' % (ds.threshold, hr, ndcg))
    # print("Training time:{}".format(end_time - start_time))
    return -hr


if __name__ == '__main__':
    # udp_file = 'data/ml-100k/udp'
    # upp_file = 'data/ml-100k/upp'
    # upp = load_dict(upp_file)
    # udp = load_dict(udp_file)
    # data_name = 'data/yahoomusic/yahoo_music'
    # data_name = 'data/ml-100k/ml-100k'
    data_name = 'data/IV/amazon'
    main(data=data_name,
         epochs=2,
         embedding_dim=10,
         regularization=0.01,
         unobserved_weight=1.0,
         stddev=0.3,
         epsilon=0.05,
         privacy_specification=None,
         user_privacy_pref=None,
         random_seed=3)

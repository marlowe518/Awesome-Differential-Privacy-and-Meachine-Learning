import numpy as np
import pandas as pd
import time
import pickle
from risk_pattern_mining import loadData


def riskDegreeComputing(mrpU, target):
    totalSum = 0
    for mrp, supp in mrpU.items():
        if target in mrp:
            totalSum += 1 / (len(mrp) * supp)
    riskDegree = totalSum
    return riskDegree


def getUserPrivacyLevel(size, eps_c=0.1, eps_m=0.2, eps_l=1.0, prob_c=0.54, prob_m=0.37, prob_l=0.09, random_seed=None):
    '''
    :param size:
    :param eps_c:
    :param eps_m:
    :param eps_l:
    :param prob_c:
    :param prob_m:
    :param prob_l:
    :param random_seed:
    :return:
    '''
    # TODO make the parameters avaliable
    if random_seed is not None:
        np.random.seed(random_seed)
    # eps_c, eps_m, eps_l = 0.1, 0.2, 1.0
    # prob_c, prob_m, prob_l = 0.54, 0.37, 0.09
    eps_c_user = np.random.uniform(eps_c, eps_m)
    eps_m_user = np.random.uniform(eps_m, eps_l)
    eps_l_user = eps_l
    return np.random.choice([eps_c_user, eps_m_user, eps_l_user], p=[prob_c, prob_m, prob_l], size=size)


def userRiskDegree(mrpU, uid, user_priv_pref):
    # we get the unique items in the mrpU
    all_items = [list(itemset) for itemset in mrpU.keys()]
    temp = sum(all_items, [])
    unique_items = list(set(temp))
    riskDegreeDict = {}
    privacyLevelDict = {}
    alpha = 0.5  # TODO set the parameters
    for item in unique_items:
        riskDegree = riskDegreeComputing(mrpU, item)
        privacyLevel = 1 / (1 + riskDegree)
        privacyLevel = alpha * user_priv_pref[uid] + (1 - alpha) * privacyLevel
        riskDegreeDict.setdefault(item, riskDegree)
        privacyLevelDict.setdefault(item, privacyLevel)
    return riskDegreeDict, privacyLevelDict


def urpToudp(urp_dict, user_priv_pref):
    users = len(urp_dict)
    udp = {}
    upp = {}
    count = 0
    print('local time: {}'.format(time.asctime(time.localtime(time.time()))))
    for user in urp_dict:
        udp[user], upp[user] = userRiskDegree(urp_dict[user]['RPs'], uid=user, user_priv_pref=user_priv_pref)
        count += 1
        print('\r' + str(user) + '/' + str(count) + '/' + str(users), end='', flush=True)
        time.sleep(.02)
    print('local time: {}'.format(time.asctime(time.localtime(time.time()))))
    return udp, upp


def transIGtoPrivacyLevel(riskDegree):
    privacyBudget = 1 / (1 + riskDegree)
    return privacyBudget


def udpToMat(udp, epsUpper, epsLower, m, n):
    '''
    :param udp: dict
    :return:
    '''
    mat = np.zeros((m, n))
    for user in udp.keys():
        for item in udp[user].keys():
            pl = transIGtoPrivacyLevel(udp[user][item])
            mat[int(user)][int(item)] = pl

    # turn the epsilon into a specific range
    rM = mat[mat > 0]
    rMmin, rMmax = np.min(rM), np.max(rM)
    k = (epsUpper - epsLower) / (rMmax - rMmin)
    new_mat = epsLower + k * (mat - rMmin)
    new_mat[mat == 0] = epsUpper
    return new_mat


# def udpToList(udp, epsUpper, epsLower, m, n):
#     return udp_list


def loadSample():
    dict1 = {'1': {'RPs': {frozenset({'1', '2'}): 1, frozenset({'2', '3'}): 1, frozenset({'3', '4'}): 1}}}
    return dict1


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # _____________________________  load data ________________________________________ #
    # trainpath = './data/ml-100k/ml-100k_train_data'
    trainpath = './data/ml-1m/ml-1m_train_data'
    # urpFile = './data/ml-100k/eva_data/urp_m_2_p_2'
    # urpFile = './data/yahoo/eva_data/urp_m_2_p_2'
    # urpFile = './data/ml-100k/urp_m_2_p_2'
    urpFile = './data/ml-1m/urp_m_2_p_2'
    # urpFile = './data/yahoo/urp'
    # udpFile = './data/udp_sample'
    # urpFile = './data/yahoomusic/urp_m_2_p_2'
    # udp = main(urpFile)
    # save_dict(udp, udpFile)
    # udp = load_dict(udpFile)
    # udp = {3: {'3': 1.0, '1': 1.0, '6': 1.0},
    #        4: {'7': 0.9182958340544894, '8': 0.9182958340544894, '1': 0.9182958340544894, '5': 0.9182958340544894},
    #        1: {'3': 0.0, '5': 0.9182958340544894, '0': 0.9182958340544894, '2': 0.9182958340544894},
    #        2: {'4': 1.0, '1': 0.0, '0': 1.0, '2': 1.0}}
    # _____________________________  compute UDP ________________________________________ #
    # get user privacy preferences
    Data = loadData(trainpath)
    user_ids = Data.keys()
    user_priv_pref = dict(zip(user_ids, getUserPrivacyLevel(len(user_ids))))

    urp = load_dict(urpFile)  # Note: not all user are included in the risk patter set
    # urp = loadSample()
    udp, upp = urpToudp(urp, user_priv_pref)

    # save user risk degree
    print(udp)
    udpFile = '/'.join(urpFile.split('/')[:-1]) + '/udp'
    save_dict(udp, udpFile)

    # save user privacy level preferences
    print(upp)
    uppFile = '/'.join(urpFile.split('/')[:-1]) + '/upp'
    save_dict(upp, uppFile)

    # save user personal privacy preferences
    upppFile = '/'.join(urpFile.split('/')[:-1]) + '/uppp'
    save_dict(user_priv_pref, upppFile)

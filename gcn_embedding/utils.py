from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import networkx as nx
from sklearn import preprocessing
from keras.utils import to_categorical
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    data_size = int(y.shape[0])
    n_train = int(0.75 * data_size)
    n_valid = int(0.1 * data_size)

    idx_train = range(n_train)
    idx_val = range(n_train, n_train + n_valid)
    idx_test = range(n_train + n_valid, data_size)
    
    y_train = np.zeros(y.shape, dtype=np.int32) #y:label
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]

    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask, val_mask, test_mask

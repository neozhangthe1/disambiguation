import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def normalize_vectors(vectors):
    scaler = StandardScaler()
    vectors_norm = scaler.fit_transform(vectors)
    return vectors_norm


def preprocess_graph(adj):  # use original version, adj not contain diags
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)
'''


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0])
    adj_dense = np.array(adj.todense())
    rowmax = np.zeros((adj.shape[0],))
    for i in range(adj.shape[0]):
        # print(max(adj_dense[i, :]))
        rowmax[i] = max(1, max(adj_dense[i, :]))
        # rowmax_orig = adj.max(axis=1).data
        # print(rowmax_orig.shape)
        # for i in range(rowmax.shape[0]):
        # rowmax[i] = max(1, rowmax_orig[i])
    # print(adj.shape, rowmax.shape)
    adj_ = adj + sp.diags(rowmax)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)
'''

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def gen_train_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    data = np.ones(edges.shape[0])
    adj_train = sp.csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train


def cal_pos_weight(adj):
    pos_edges_num = adj.nnz
    return (adj.shape[0] * adj.shape[0] - pos_edges_num) / pos_edges_num

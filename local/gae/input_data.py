from os.path import join
import numpy as np
import scipy.sparse as sp
from utils import settings

local_na_dir = join(settings.DATA_DIR, 'local', 'na')


def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))


def load_local_data(path=local_na_dir, name='cheng_cheng'):
    # Load local paper network dataset
    print('Loading {} dataset...'.format(name))

    idx_features_labels = np.genfromtxt(join(path, "{}_pubs_content.txt".format(name)), dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)  # sparse?
    labels = encode_labels(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(join(path, "{}_pubs_network.txt".format(name)), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return adj, features, labels
"""


def load_local_data(path=local_na_dir, name="cheng_cheng"):
    # Load citation network dataset (cora only for now)
    print('Loading {} dataset...'.format(name))

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    idx_features_labels = np.genfromtxt(join(path, "{}_pubs_content.txt".format(name)), dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_labels(idx_features_labels[:, -1])

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges_unordered = np.genfromtxt(join(path, "{}_pubs_network.txt".format(name)), dtype=np.dtype(str))
    edges_unordered_copy = []
    if edges_unordered.shape[1] > 2:
        weights = np.array(edges_unordered[:, -1], dtype=np.float32)
    else:
        weights = np.ones(edges_unordered.shape[0])
    # edges_unordered = np.array(edges_unordered[:, :-1], dtype=np.int64)
    edges_unordered = np.array(edges_unordered[:, :2], dtype=np.str)
    for i, edge in enumerate(edges_unordered):
        if edge[0] in idx_map and edge[1] in idx_map:
            edges_unordered_copy.append(edge)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return adj, features, labels
"""


if __name__ == '__main__':
    load_local_data(name='zhigang_zeng')

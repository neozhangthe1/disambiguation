from __future__ import division
from __future__ import print_function

import os
import time
from os.path import join

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from local.gae.optimizer import OptimizerAE, OptimizerVAE
from local.gae.input_data import load_local_data
from local.gae.model import GCNModelAE, GCNModelVAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
# from gae.utils.vis import tsne, plot_scatters, draw_graph
from utils.cluster import clustering
from utils.data_utils import load_json
from utils.eval import pairwise_precision_recall_f1
from utils import settings

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()


def gae_for_na(name):
    adj, features, labels = load_local_data(name=name)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)

    node_sim = np.dot(features, features.transpose())
    node_sim = (node_sim >= 1).astype(np.int)
    adj_train = sp.csr_matrix(adj_train + node_sim)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    input_feature_dim = features.shape[1]
    if FLAGS.is_sparse:  # TODO to test
        # features = sparse_to_tuple(features.tocoo())
        # features_nonzero = features[1].shape[0]
        features = features.todense()  # TODO incorrect
    else:
        features = normalize_vectors(features)

    # Define placeholders
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, input_feature_dim)),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model
    model = None
    if model_str == 'gcn_ae':
        # model = GCNModelAE(placeholders, num_features, features_nonzero)
        model = GCNModelAE(placeholders, input_feature_dim)
    elif model_str == 'gcn_vae':
        # model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
        model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    print('positive edge weight', pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        '''
        if model_str == 'gcn_ae':
            emb = sess.run(model.z_mean, feed_dict=feed_dict)
        elif model_str == 'gcn_vae':
            emb = sess.run(model.z, feed_dict=feed_dict)
        '''
        return emb

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "time=", "{:.5f}".format(time.time() - t))

    emb = get_embs()
    n_clusters = len(set(labels))
    emb_norm = normalize_vectors(emb)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))
    return [prec, rec, f1], num_nodes, n_clusters


def load_test_names():
    return load_json(settings.DATA_DIR, 'test_name_list.json')


def main():
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')
    metrics = np.zeros(3)
    cnt = 0
    for name in names:
        cur_metric, num_nodes, n_clusters = gae_for_na(name)
        wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}\n'.format(
            name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2]))
        wf.flush()
        for i, m in enumerate(cur_metric):
            metrics[i] += m
        cnt += 1
        print('average until now', metrics / cnt)
        time_acc = time.time()-start_time
        print(cnt, 'names', time_acc, 'avg time', time_acc/cnt)
    metrics = metrics/len(names)
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        metrics[0], metrics[1], metrics[2]))
    wf.close()


if __name__ == '__main__':
    gae_for_na('hongbin_liang')
    # main()

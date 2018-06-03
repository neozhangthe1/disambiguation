from os.path import join
import os
import codecs
import json
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, concatenate, Input, merge, Lambda
from keras.optimizers import Adagrad, Adam
from global_.triplet import l2Norm, euclidean_distance, triplet_loss, accuracy
from utils import eval_utils
from utils import data_utils
from utils import settings


class GlobalTripletModel:

    def __init__(self, data_scale):
        self.data_scale = data_scale
        self.train_triplets_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.data_scale))
        self.test_triplets_dir = join(settings.OUT_DIR, 'test-triplets')
        self.train_triplet_files_num = self.get_triplets_files_num(self.train_triplets_dir)
        self.test_triplet_files_num = self.get_triplets_files_num(self.test_triplets_dir)
        print('test file num', self.test_triplet_files_num)

    @staticmethod
    def get_triplets_files_num(path_dir):
        files = []
        for f in os.listdir(path_dir):
            if f.startswith('anchor_embs_'):
                files.append(f)
        return len(files)

    def load_batch_triplets(self, f_idx, role='train'):
        if role == 'train':
            cur_dir = self.train_triplets_dir
        else:
            cur_dir = self.test_triplets_dir
        X1 = data_utils.load_data(cur_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
        X2 = data_utils.load_data(cur_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
        X3 = data_utils.load_data(cur_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        return X1, X2, X3

    def load_triplets_data(self, role='train'):
        X1 = np.empty([0, 100])
        X2 = np.empty([0, 100])
        X3 = np.empty([0, 100])
        if role == 'train':
            f_num = self.train_triplet_files_num
        else:
            f_num = self.test_triplet_files_num
        for i in range(f_num):
            print('load', i)
            x1_batch, x2_batch, x3_batch = self.load_batch_triplets(i, role)
            p = np.random.permutation(len(x1_batch))
            x1_batch = np.array(x1_batch)[p]
            x2_batch = np.array(x2_batch)[p]
            x3_batch = np.array(x3_batch)[p]
            X1 = np.concatenate((X1, x1_batch))
            X2 = np.concatenate((X2, x2_batch))
            X3 = np.concatenate((X3, x3_batch))
        return X1, X2, X3

    @staticmethod
    def create_triplet_model():
        emb_anchor = Input(shape=(100, ), name='anchor_input')
        emb_pos = Input(shape=(100, ), name='pos_input')
        emb_neg = Input(shape=(100, ), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

        inter_layer = Model(inputs=model.get_input_at(0), outputs=model.get_layer('norm_layer').get_output_at(0))

        return model, inter_layer

    def load_triplets_model(self):
        model_dir = join(settings.OUT_DIR, 'model')
        rf = open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'r')
        model_json = rf.read()
        rf.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))
        return loaded_model

    def train_triplets_model(self):
        X1, X2, X3 = self.load_triplets_data()
        n_triplets = len(X1)
        print('loaded')
        model, inter_model = self.create_triplet_model()
        # print(model.summary())

        '''
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.concatenate((X1, X2, X3)))
        X_anchor = X_scaled[: n_triplets]
        X_pos = X_scaled[n_triplets: 2*n_triplets]
        X_neg = X_scaled[2*n_triplets:]
        '''

        X_anchor, X_pos, X_neg = X1, X2, X3
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}
        model.fit(X, np.ones((n_triplets, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)

        model_json = model.to_json()
        model_dir = join(settings.OUT_DIR, 'model')
        os.makedirs(model_dir, exist_ok=True)
        with open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'w') as wf:
            wf.write(model_json)
        model.save_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))

        test_triplets = self.load_triplets_data(role='test')
        auc_score = eval_utils.full_auc(model, test_triplets)
        # print('AUC', auc_score)

        loaded_model = self.load_triplets_model()
        print('triplets model loaded')
        auc_score = eval_utils.full_auc(loaded_model, test_triplets)


if __name__ == '__main__':
    global_model = GlobalTripletModel(data_scale=1000000)
    global_model.train_triplets_model()
    print('done')
    

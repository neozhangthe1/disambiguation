import logging
from os.path import join
from gensim.models import Word2Vec
import numpy as np
import random
from utils import data_utils
from utils import feature_utils
from utils import settings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
EMB_DIM = 50


class EmbeddingModel(object):

    def __init__(self, name="scopus"):
        self.model = None
        self.name = name

    @staticmethod
    def train(wf_name, size=EMB_DIM):
        data = []
        for i, paper in enumerate(data_utils.pubs_load_generator()):
            if i % 100 == 0:
                print('paper cnt', i)
            for i, a in enumerate(paper.get('authors', [])):
                author_feature = feature_utils.extract_author_features(paper, i)
                random.shuffle(author_feature)
                print(author_feature)
                data.append(author_feature)
        model = Word2Vec(
            data, size=size, window=50, min_count=5, workers=20,
        )
        model.save(join(settings.EMB_DIR, '{}.emb'.format(wf_name)))

    @staticmethod
    def load(name):
        e = EmbeddingModel()
        e.model = Word2Vec.load(join(settings.EMB_DIR, '{}.emb'.format(name)))
        return e

    def project_embedding(self, tokens, idf=None):
        vectors = []
        sum_weight = 0
        for token in tokens:
            if not token in self.model.wv:
                continue
            weight = 1
            if idf and token in idf:
                weight = idf[token]
            v = self.model.wv[token] * weight
            vectors.append(v)
            sum_weight += weight
        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        emb = np.sum(vectors, axis=0)
        # emb /= sum_weight
        return emb


if __name__ == '__main__':
    # rf_path = join(settings.DATA_DIR, 'global', 'author_features.txt')
    wf_name = 'scopus'
    EmbeddingModel.train(wf_name)
    # emb_model = EmbeddingModel.load('scopus')
    print('loaded')

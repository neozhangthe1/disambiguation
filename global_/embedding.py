import logging
from os.path import join
from gensim.models import Word2Vec
import numpy as np
import random
from utils.cache import LMDBClient
from utils import data_utils
from utils.data_utils import Singleton
from utils import settings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
EMB_DIM = 100


@Singleton
class EmbeddingModel:

    def __init__(self, name="scopus"):
        self.model = None
        self.name = name

    def train(self, wf_name, size=EMB_DIM):
        data = []
        LMDB_NAME = 'pub_authors.feature'
        lc = LMDBClient(LMDB_NAME)
        author_cnt = 0
        with lc.db.begin() as txn:
            for k in txn.cursor():
                author_feature = data_utils.deserialize_embedding(k[1])
                if author_cnt % 10000 == 0:
                    print(author_cnt, author_feature[0])
                author_cnt += 1
                random.shuffle(author_feature)
                # print(author_feature)
                data.append(author_feature)
        self.model = Word2Vec(
            data, size=size, window=5, min_count=5, workers=20,
        )
        self.model.save(join(settings.EMB_DATA_DIR, '{}.emb'.format(wf_name)))

    def load(self, name):
        self.model = Word2Vec.load(join(settings.EMB_DATA_DIR, '{}.emb'.format(name)))
        return self.model

    def project_embedding(self, tokens, idf=None):
        """
        weighted average of token embeddings
        :param tokens: input words
        :param idf: IDF dictionary
        :return: obtained weighted-average embedding
        """
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
            print('all tokens not in w2v models')
            # return np.zeros(self.model.vector_size)
            return None
        emb = np.sum(vectors, axis=0)
        emb /= sum_weight
        return emb


if __name__ == '__main__':
    wf_name = 'scopus'
    emb_model = EmbeddingModel.Instance()
    emb_model.train(wf_name)
    print('loaded')

from os.path import join
import codecs
import math
from collections import defaultdict as dd
from global_.embedding import EmbeddingModel
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import settings

start_time = datetime.now()


def dump_author_features_to_file():
    pubs_train = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'pubs_raw_train.pkl')
    pubs_test = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'pubs_raw_test.pkl')
    pubs_dict = {**pubs_test, **pubs_train}
    print('n_papers', len(pubs_dict))
    wf = codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features.txt'), 'w', encoding='utf-8')
    for i, pid in enumerate(pubs_dict):
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        paper = pubs_dict[pid]
        if "title" not in paper or "authors" not in paper:
            continue
        if len(paper["authors"]) > 30:
            print(i, paper["sid"], len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        n_authors = len(paper.get('authors', []))
        for j in range(n_authors):
            author_feature = feature_utils.extract_author_features(paper, j)
            aid = '{}-{}'.format(paper['sid'], j)
            wf.write(aid + '\t' + ' '.join(author_feature) + '\n')
    wf.close()


def dump_author_features_to_cache():
    LMDB_NAME = 'pub_authors.feature'
    lc = LMDBClient(LMDB_NAME)
    with codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features.txt'), 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            if i % 1000 == 0:
                print('line', i)
            items = line.rstrip().split('\t')
            pid_order = items[0]
            author_features = items[1].split()
            lc.set(pid_order, author_features)


def cal_feature_idf():
    feature_dir = join(settings.DATA_DIR, 'global')
    counter = dd(int)
    cnt = 0
    LMDB_NAME = 'pub_authors.feature'
    lc = LMDBClient(LMDB_NAME)
    author_cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor():
            features = data_utils.deserialize_embedding(k[1])
            if author_cnt % 10000 == 0:
                print(author_cnt, features[0], counter.get(features[0]))
            author_cnt += 1
            for f in features:
                cnt += 1
                counter[f] += 1
    idf = {}
    for k in counter:
        idf[k] = math.log(cnt / counter[k])
    data_utils.dump_data(dict(idf), feature_dir, "feature_idf.pkl")


def dump_author_embs():
    emb_model = EmbeddingModel.Instance()
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
    print('idf loaded')
    LMDB_NAME_FEATURE = 'pub_authors.feature'
    lc_feature = LMDBClient(LMDB_NAME_FEATURE)
    LMDB_NAME_EMB = "author_100.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    cnt = 0
    with lc_feature.db.begin() as txn:
        for k in txn.cursor():
            if cnt % 1000 == 0:
                print('cnt', cnt, datetime.now()-start_time)
            cnt += 1
            pid_order = k[0].decode('utf-8')
            features = data_utils.deserialize_embedding(k[1])
            cur_emb = emb_model.project_embedding(features, idf)
            if cur_emb is not None:
                lc_emb.set(pid_order, cur_emb)


if __name__ == '__main__':
    dump_author_features_to_file()
    dump_author_features_to_cache()
    emb_model = EmbeddingModel.Instance()
    emb_model.train('scopus')
    cal_feature_idf()
    dump_author_embs()
    print('done', datetime.now()-start_time)

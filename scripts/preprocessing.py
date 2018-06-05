from os.path import join
import codecs
import math
import multiprocessing as mp
from collections import defaultdict as dd
from global_.embedding import EmbeddingModel
from datetime import datetime
from copy import deepcopy
from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import settings

global_dir = join(settings.DATA_DIR, 'global')
start_time = datetime.now()


def put_papers(task_q, N_PROC):
    for i, paper in enumerate(data_utils.pubs_load_generator()):
        n_authors = len(paper.get('authors', []))
        for j in range(n_authors):
            task_q.put((paper, j))
    for _ in range(N_PROC):
        task_q.put((None, None))


def cal_author_features(task_q, feature_q):
    cnt = 0
    while True:
        paper, order = task_q.get()
        if paper is None:
            feature_q.put((None, None))
            break
        cnt += 1
        author_feature = feature_utils.extract_author_features(paper, order)
        feature_q.put(('{}-{}'.format(paper['sid'], order), author_feature))


def dump_author_features_to_cache():
    LMDB_NAME = 'pub_authors.feature'
    lc = LMDBClient(LMDB_NAME)
    with codecs.open(join(global_dir, 'author_features.txt'), 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            if i % 1000 == 0:
                print('line', i)
            items = line.rstrip().split('\t')
            pid_order = items[0]
            author_features = items[1].split()
            lc.set(pid_order, author_features)
    '''
    N_PROC = 50

    task_q = mp.Queue(1000)
    feature_q = mp.Queue(1000)

    producer_p = mp.Process(target=put_papers, args=(task_q, N_PROC))
    consumer_ps = [mp.Process(target=cal_author_features, args=(task_q, feature_q)) for _ in range(N_PROC)]
    producer_p.start()
    [p.start() for p in consumer_ps]

    cnt = 0
    LMDB_NAME = 'pub_authors_test.feature'
    lc = LMDBClient(LMDB_NAME)

    st = datetime.now()

    while True:
        if cnt % 100 == 0:
            et = datetime.now()
            print('paper cnt', cnt, 'timedelta', et - st, 'total time', datetime.now()-start_time)
            st = deepcopy(et)
        pid_order, feature = feature_q.get()
        if pid_order is None:
            break
        lc.set(pid_order, feature)
        cnt += 1
    '''


def dump_author_features_to_file():
    pubs_train = data_utils.load_data(global_dir, 'pubs_raw_train.pkl')
    pubs_test = data_utils.load_data(global_dir, 'pubs_raw_test.pkl')
    pubs_dict = {**pubs_test, **pubs_train}
    print('n_papers', len(pubs_dict))
    wf = codecs.open(join(global_dir, 'author_features.txt'), 'w', encoding='utf-8')
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
    emb_model = EmbeddingModel.load('scopus')
    idf = data_utils.load_data(global_dir, 'feature_idf.pkl')
    print('idf loaded')
    LMDB_NAME_FEATURE = 'pub_authors.feature'
    lc_feature = LMDBClient(LMDB_NAME_FEATURE)
    LMDB_NAME_EMB = "author_100.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    # N_PROC = 50
    # task_q = mp.Queue(1000)
    # emb_q = mp.Queue(1000)
    # consumer_ps = [mp.Process(target=emb_model.project_embedding, args=(task_q, emb_q)) for _ in range(N_PROC)]
    # [p.start() for p in consumer_ps]
    cnt = 0
    with lc_feature.db.begin() as txn:
        for k in txn.cursor():
            if cnt % 100 == 0:
                print('cnt', cnt)
            cnt += 1
            pid_order = k[0].decode('utf-8')
            print(pid_order)
            features = data_utils.deserialize_embedding(k[1])
            print(features)
            lc_emb.set(pid_order, emb_model.project_embedding(features, idf))
    '''
    for paper in data_utils.pubs_load_generator():
        if not "title" in paper or not "authors" in paper:
            continue
        if len(paper["authors"]) > 30:
            print(cnt, paper["sid"], len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        if cnt % 100 == 0:
            print('paper count', cnt, paper["sid"], len(paper["authors"]))
        cnt += 1
        for i, author in enumerate(paper.get('authors', [])):
            # author_feature = feature_utils.extract_author_features(paper, i)
            author_feature = lc_feature.get('{}-{}'.format(paper['sid'], i))
            # TODO
            task_q.put()
            lc_emb.set("%s-%s" % (paper["sid"], i), emb_model.project_embedding(author_feature, idf))
    '''


if __name__ == '__main__':
    # dump_author_features_to_file()
    cal_feature_idf()
    # dump_author_embs()
    print('done', datetime.now()-start_time)

from os.path import join
import multiprocessing as mp
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


def dump_author_features():
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


def dump_author_embs():
    emb_model = EmbeddingModel.load('scopus')
    cnt = 0
    idf = data_utils.load_data(global_dir, 'feature_idf.pkl')
    print('idf loaded')
    LMDB_NAME = "author_100.emb.weighted"
    lc = LMDBClient(LMDB_NAME)
    for paper in data_utils.pubs_load_generator():
        if not "title" in paper or not "authors" in paper:
            continue
        if len(paper["authors"]) > 30:
            print(cnt, paper["sid"], len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        if cnt % 1000 == 0:
            print(cnt, paper["sid"], len(paper["authors"]))
        cnt += 1
        for i, author in enumerate(paper.get('authors', [])):
            author_feature = feature_utils.extract_author_features(paper, i)
            lc.set("%s-%s" % (paper["sid"], i), emb_model.project_embedding(author_feature, idf))


if __name__ == '__main__':
    dump_author_features()
    # dump_author_embs()
    print('done', datetime.now()-start_time)

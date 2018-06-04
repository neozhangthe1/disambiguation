from os.path import join
from global_.embedding import EmbeddingModel
from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import settings

global_dir = join(settings.DATA_DIR, 'global')


def pubs_load_generator():
    name_to_pubs_train = data_utils.load_data(global_dir, 'pubs_raw_train.pkl')
    name_to_pubs_test = data_utils.load_data(global_dir, 'pubs_raw_test.pkl')
    name_to_pubs = {**name_to_pubs_test, **name_to_pubs_train}
    for pid in name_to_pubs:
        yield name_to_pubs[pid]


def dump_author_features_lmdb():
    emb_model = EmbeddingModel.load('scopus')
    cnt = 0
    idf = data_utils.load_data(global_dir, 'feature_idf.pkl')
    LMDB_NAME = "author_100.emb.weighted"
    lc = LMDBClient(LMDB_NAME)
    for paper in pubs_load_generator():
        if not "title" in paper or not "authors" in paper:
            continue
        if len(paper["authors"]) > 30:
            print(cnt, paper["sid"], len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        if cnt % 1000 == 0:
            print(cnt, paper["sid"], len(paper["authors"]))
        cnt += 1
        author_features = feature_utils.extract_author_features(paper)
        for i, f in enumerate(author_features):
            lc.set("%s-%s" % (paper["sid"], i), emb_model.project_embedding(f, idf))


if __name__ == '__main__':
    dump_author_features_lmdb()

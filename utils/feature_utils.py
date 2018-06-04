from os.path import join
from collections import defaultdict as dd
import math
from multiprocessing import Pool
from datetime import datetime
from itertools import chain
from utils.cache import LMDBClient
from utils import string_utils
from utils import data_utils
from utils import settings


def transform_feature(data, f_name, k=1):
    if type(data) is str:
        data = data.split()
    assert type(data) is list
    features = []
    for d in data:
        features.append("__%s__%s" % (f_name.upper(), d))
    return features


def extract_common_features(item):
    title_features = transform_feature(string_utils.clean_sentence(item["title"], stemming=True).lower(), "title")
    keywords_features = []
    keywords = item.get("keywords", "")
    if len(keywords) > 3:
        keywords_features = transform_feature([string_utils.clean_name(k) for k in item["keywords"].split(";")], 'keyword')
    venue_features = []
    venue_name = item.get('venue', {}).get("raw", "")
    if len(venue_name) > 2:
        venue_features = transform_feature(string_utils.clean_sentence(venue_name.lower()), "venue")
    return title_features, keywords_features, venue_features


def extract_author_features(item, order=None):
    title_features, keywords_features, venue_features = extract_common_features(item)
    author_features = []
    for i, author in enumerate(item["authors"]):
        if order and i != order:
            continue
        name_feature = []
        org_features = []
        org_name = string_utils.clean_name(author.get("org", ""))
        if len(org_name) > 2:
            org_features.extend(transform_feature(org_name, "org"))
        for j, coauthor in enumerate(item["authors"]):
            if i == j:
                continue
            coauthor_name = coauthor.get("name", "")
            coauthor_org = string_utils.clean_name(coauthor.get("org", ""))
            if len(coauthor_name) > 2:
                name_feature.extend(
                    transform_feature([string_utils.clean_name(coauthor_name)], "name")
                )
            if len(coauthor_org) > 2:
                org_features.extend(
                    transform_feature(string_utils.clean_sentence(coauthor_org.lower()), "org")
                )
        author_features.append(
            name_feature + org_features + title_features + keywords_features + venue_features
        )
    author_features = list(chain.from_iterable(author_features))
    return author_features


def dump_batch_author_features(paper):
    LMDB_NAME = 'pub_authors.feature'
    lc = LMDBClient(LMDB_NAME)
    n_authors = len(paper.get('authors', []))
    author_features = [extract_author_features(paper, j) for j in range(n_authors)]
    for j in range(n_authors):
        lc.set('{}-{}'.format(paper['sid'], j), author_features[j])


def dump_author_features():
    pool = Pool(32)
    batch_papers = []
    start_time = datetime.now()
    for i, paper in enumerate(data_utils.pubs_load_generator()):
        if i % 100 == 0:
            print('paper cnt', i, datetime.now()-start_time)
        if len(batch_papers) % 100 == 0:
            pool.map(dump_batch_author_features, batch_papers)
            batch_papers = []
        batch_papers.append(paper)
    pool.map(dump_batch_author_features, batch_papers)
    print('done')


def cal_feature_idf():
    feature_dir = join(settings.DATA_DIR, 'global_')
    counter = dd(int)
    cnt = 0
    for line in open(join(feature_dir, 'author_features.txt')):
        x = line.split("\t")
        if cnt % 1000 == 0:
            print(cnt, x[0], len(counter))
        for f in set(x[1].split()):
            counter[f] += 1
        cnt += 1
    idf = {}
    for k in counter:
        idf[k] = math.log(cnt / counter[k])
    data_utils.dump_data(dict(idf), feature_dir, "feature_idf.pkl")


if __name__ == '__main__':
    # cal_feature_idf()
    dump_author_features()

from multiprocessing import Pool
from datetime import datetime
from itertools import chain
from utils.cache import LMDBClient
from utils import string_utils
from utils import data_utils


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
    keywords = item.get("keywords")
    if keywords:
        keywords_features = transform_feature([string_utils.clean_name(k) for k in keywords], 'keyword')
    venue_features = []
    venue_name = item.get('venue', '')
    if len(venue_name) > 2:
        venue_features = transform_feature(string_utils.clean_sentence(venue_name.lower()), "venue")
    return title_features, keywords_features, venue_features


def extract_author_features(item, order=None):
    title_features, keywords_features, venue_features = extract_common_features(item)
    author_features = []
    for i, author in enumerate(item["authors"]):
        if order is not None and i != order:
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

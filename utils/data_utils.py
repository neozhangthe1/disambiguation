import codecs
import json
from os.path import join
import pickle
import os
from utils import settings

global_dir = join(settings.DATA_DIR, 'global')


def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


def embedding_loader(path):
    for line in open(path):
        x = line.strip().split('\t')  # after id
        yield x[1]


def pubs_load_generator():
    pubs_train = load_data(global_dir, 'pubs_raw_train.pkl')
    pubs_test = load_data(global_dir, 'pubs_raw_test.pkl')
    pubs_dict = {**pubs_test, **pubs_train}
    for pid in pubs_dict:
        yield pubs_dict[pid]

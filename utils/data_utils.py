import codecs
import json
from os.path import join
import pickle
import os


def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def embedding_loader(path):
    for line in open(path):
        x = line.strip().split('\t')  # after id
        yield x[1]

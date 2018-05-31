from os.path import join
from collections import defaultdict as dd
import math
from utils import data_utils
from utils import settings


def cal_feature_idf():
    feature_dir = join(settings.DATA_DIR, 'global')
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
    cal_feature_idf()

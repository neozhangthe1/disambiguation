from os.path import join
import codecs
import json
import numpy as np
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings


def dump_inter_emb():
    LMDB_NAME = "scopus_author_100.emb.weighted"
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'scopus_author_triplets.emb'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    global_model = GlobalTripletModel(data_scale=5000000)
    trained_global_model = global_model.load_triplets_model()
    global_dir = join(settings.DATA_DIR, 'global')
    name_to_pubs_test = data_utils.load_data(global_dir, 'name_to_pubs_test.pkl')
    for name in name_to_pubs_test:
        print('name', name)
        name_data = name_to_pubs_test[name]
        embs_input = []
        pids = []
        for i, sid in enumerate(name_data.keys()):
            if len(name_data[sid]) < 5:  # n_pubs of current author is too small
                continue
            for year, pid in name_data[sid]:
                cur_emb = lc_input.get(pid)
                if cur_emb is None:
                    continue
                embs_input.append(cur_emb)
                pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid_ in enumerate(pids):
            lc_inter.set(pid_, inter_embs)


if __name__ == '__main__':
    dump_inter_emb()
    print('done')

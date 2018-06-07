from os.path import join
import os
import numpy as np
from numpy.random import shuffle
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings


def dump_inter_emb():
    LMDB_NAME = "author_100.emb.weighted"
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    global_model = GlobalTripletModel(data_scale=1000000)
    trained_global_model = global_model.load_triplets_model()
    global_dir = join(settings.DATA_DIR, 'global')
    name_to_pubs_test = data_utils.load_data(global_dir, 'name_to_pubs_test_100.pkl')
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
            lc_inter.set(pid_, inter_embs[i])


def gen_local_data(idf_threshold=10):
    global_dir = join(settings.DATA_DIR, 'global')
    name_to_pubs_test = data_utils.load_data(global_dir, 'name_to_pubs_test_100.pkl')
    # INTER_LMDB_NAME = 'author_triplets.emb'
    INTER_LMDB_NAME = 'author_100.emb.weighted'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors.feature"
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
    idf = data_utils.load_data(global_dir, 'feature_idf.pkl')
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    # graph_dir = join(settings.DATA_DIR, 'graph-orig-emb-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        print(i, name)
        cur_person_dict = name_to_pubs_test[name]
        pids_set = set()
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for i, sid in enumerate(cur_person_dict):
            items = cur_person_dict[sid]
            if len(items) < 5:
                continue
            for year, pid in items:
                pids2label[pid] = sid
                pids.append(pid)
        shuffle(pids)
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
        wf_content.close()

        # generate network
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w')
        for i in range(n_pubs-1):
            if i % 10 == 0:
                print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i+1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    wf_network.write('{}\t{}\n'.format(pids[i], pids[j]))
        wf_network.close()


if __name__ == '__main__':
    dump_inter_emb()
    gen_local_data(idf_threshold=20)
    print('done')

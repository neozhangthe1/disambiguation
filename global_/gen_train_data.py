from os.path import join
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

LMDB_NAME = "author_100.emb.weighted"  # name consistent
lc = LMDBClient(LMDB_NAME)
start_time = datetime.now()


class TripletsGenerator:
    name2pubs_train = {}
    name2pubs_test = {}
    names_train = None
    names_test = None
    n_pubs_train = None
    n_pubs_test = None
    pids_train = []
    pids_test = []
    n_triplets = 0
    batch_size = 100000
    global_dir = join(settings.DATA_DIR, 'global')

    def __init__(self, train_scale=10000):
        self.prepare_data()
        self.save_size = train_scale

    def prepare_data(self):
        self.name2pubs_train = data_utils.load_data(self.global_dir, 'name_to_pubs_train_500.pkl')  # for test
        self.name2pubs_test = data_utils.load_data(self.global_dir, 'name_to_pubs_test_100.pkl')
        self.names_train = self.name2pubs_train.keys()
        print('names train', len(self.names_train))
        self.names_test = self.name2pubs_test.keys()
        print('names test', len(self.names_test))
        assert not set(self.names_train).intersection(set(self.names_test))
        for name in self.names_train:
            name_pubs_dict = self.name2pubs_train[name]
            for sid in name_pubs_dict:
                self.pids_train += [item[1] for item in name_pubs_dict[sid]]
        random.shuffle(self.pids_train)
        self.n_pubs_train = len(self.pids_train)
        print('pubs2train', self.n_pubs_train)

        for name in self.names_test:
            name_pubs_dict = self.name2pubs_test[name]
            for sid in name_pubs_dict:
                self.pids_test += [item[1] for item in name_pubs_dict[sid]]
        random.shuffle(self.pids_test)
        self.n_pubs_test = len(self.pids_test)
        print('pubs2test', self.n_pubs_test)

    def gen_neg_pid(self, not_in_pids, role='train'):
        if role == 'train':
            sample_from_pids = self.pids_train
        else:
            sample_from_pids = self.pids_test
        while True:
            idx = random.randint(0, len(sample_from_pids)-1)
            pid = sample_from_pids[idx]
            if pid not in not_in_pids:
                return pid

    def sample_triplet_ids(self, task_q, role='train', N_PROC=8):
        n_sample_triplets = 0
        if role == 'train':
            names = self.names_train
            name2pubs = self.name2pubs_train
        else:  # test
            names = self.names_test
            name2pubs = self.name2pubs_test
            self.save_size = 200000  # test save size
        for name in names:
            name_pubs_dict = name2pubs[name]
            for sid in name_pubs_dict:
                pub_items = name_pubs_dict[sid]
                if len(pub_items) == 1:
                    continue
                pids = [item[1] for item in pub_items]
                cur_n_pubs = len(pids)
                random.shuffle(pids)
                for i in range(cur_n_pubs):
                    pid1 = pids[i]  # pid

                    # batch samples
                    n_samples_anchor = min(6, cur_n_pubs)
                    idx_pos = random.sample(range(cur_n_pubs), n_samples_anchor)
                    for ii, i_pos in enumerate(idx_pos):
                        if i_pos != i:
                            if n_sample_triplets % 100 == 0:
                                # print('sampled triplet ids', n_sample_triplets)
                                pass
                            pid_pos = pids[i_pos]
                            pid_neg = self.gen_neg_pid(pids, role)
                            n_sample_triplets += 1
                            task_q.put((pid1, pid_pos, pid_neg))

                            if n_sample_triplets >= self.save_size:
                                for j in range(N_PROC):
                                    task_q.put((None, None, None))
                                return
        for j in range(N_PROC):
            task_q.put((None, None, None))
        print('here1')

    def gen_emb_mp(self, task_q, emb_q):
        while True:
            pid1, pid_pos, pid_neg = task_q.get()
            if pid1 is None:
                break
            emb1 = lc.get(pid1)
            emb_pos = lc.get(pid_pos)
            emb_neg = lc.get(pid_neg)
            if emb1 is not None and emb_pos is not None and emb_neg is not None:
                emb_q.put((emb1, emb_pos, emb_neg))
        emb_q.put((False, False, False))
        print('here2')

    def gen_triplets_mp(self, role='train'):
        N_PROC = 8

        task_q = mp.Queue(N_PROC * 6)
        emb_q = mp.Queue(1000)

        producer_p = mp.Process(target=self.sample_triplet_ids, args=(task_q, role, N_PROC))
        consumer_ps = [mp.Process(target=self.gen_emb_mp, args=(task_q, emb_q)) for _ in range(N_PROC)]
        producer_p.start()
        [p.start() for p in consumer_ps]

        cnt = 0

        while True:
            if cnt % 1000 == 0:
                print('get', cnt, datetime.now()-start_time)
            emb1, emb_pos, emb_neg = emb_q.get()
            if emb1 is False:
                print('here3')
                producer_p.terminate()
                producer_p.join()
                [p.terminate() for p in consumer_ps]
                [p.join() for p in consumer_ps]
                print('here4')
                break
            cnt += 1
            yield (emb1, emb_pos, emb_neg)

    def dump_triplets(self, role='train'):
        triplets = self.gen_triplets_mp(role)
        if role == 'train':
            out_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.save_size))
        else:
            out_dir = join(settings.OUT_DIR, 'test-triplets')
        os.makedirs(out_dir, exist_ok=True)
        anchor_embs = []
        pos_embs = []
        neg_embs = []
        f_idx = 0
        for i, t in enumerate(triplets):
            if i % 100 == 0:
                print(i, datetime.now()-start_time)
            emb_anc, emb_pos, emb_neg = t[0], t[1], t[2]
            anchor_embs.append(emb_anc)
            pos_embs.append(emb_pos)
            neg_embs.append(emb_neg)
            if len(anchor_embs) == self.batch_size:
                data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
                f_idx += 1
                anchor_embs = []
                pos_embs = []
                neg_embs = []
        if anchor_embs:
            data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        print('dumped')


if __name__ == '__main__':
    data_gen = TripletsGenerator(train_scale=1000000)
    data_gen.dump_triplets(role='train')
    data_gen.dump_triplets(role='test')

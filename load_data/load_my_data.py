#!/usr/bin/env python3

import time
import numpy as np
import torch
import scipy.sparse as sp

class AlignmentData:

    def __init__(self, data_dir="Mydata", file_num=2,rate=0.3, share=False, swap=False, val=0.0, with_r=False, device = torch.device('cpu')):
        t_ = time.time()



        self.rate = rate
        self.val = val
        self.file_num = file_num
        if file_num==4:
            self.ins2id_dict, self.id2ins_dict, [self.kg1_ins_ids, self.kg2_ins_ids, self.kg3_ins_ids,
                                                 self.kg4_ins_ids, ] = self.load_dict(data_dir + "/ent_ids_",
                                                                                      file_num=file_num)
            self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids, self.kg3_rel_ids,
                                                 self.kg4_rel_ids] = self.load_dict(
                data_dir + "/rel_ids_", file_num=file_num)
        if file_num==3:
            self.ins2id_dict, self.id2ins_dict, [self.kg1_ins_ids, self.kg2_ins_ids,self.kg3_ins_ids] = self.load_dict(data_dir + "/ent_ids_", file_num=file_num)
            self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids,self.kg3_rel_ids] = self.load_dict(
            data_dir + "/rel_ids_", file_num=file_num)
        self.ins_num = len(self.ins2id_dict)
        self.rel_num = len(self.rel2id_dict)
        self.triple_idx = self.load_triples(data_dir + "/triples_", file_num=file_num)
        self.ill_idx = self.load_label(data_dir + "/label",file_num=file_num)
        np.random.shuffle(self.ill_idx)
        self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = np.array(
            self.ill_idx[:int(len(self.ill_idx) // 1 * rate)], dtype=np.int32), np.array(
            self.ill_idx[int(len(self.ill_idx) // 1 * rate): int(len(self.ill_idx) // 1 * (rate + val))],
            dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * (rate + val)):], dtype=np.int32)
        self.ins_G_edges_idx, self.ins_G_values_idx, self.r_ij_idx = self.gen_sparse_graph_from_triples(self.triple_idx,
                                                                                                        self.ins_num,
                                                                                                        with_r)

        assert (share != swap or (share == False and swap == False))
        if share:
            self.triple_idx = self.share(self.triple_idx, self.ill_train_idx)  # 1 -> 2:base
            self.kg1_ins_ids = (self.kg1_ins_ids - set(self.ill_train_idx[:, 0])) | set(self.ill_train_idx[:, 1])
            self.ill_train_idx = []
        if swap:
            self.triple_idx = self.swap(self.triple_idx, self.ill_train_idx)
        self.labeled_alignment = set()
        self.boot_triple_idx = []
        self.boot_pair_dix = []
        self.device = device

       


        self.init_time = time.time() - t_

    def load_triples(self, data_dir, file_num):
        if file_num == 4:
            file_names = [data_dir + str(i)+".txt" for i in range(1, 5)]
        elif file_num == 3:
            file_names = [data_dir + str(i)+".txt" for i in range(1, 4)]
        else:
            file_names = [data_dir+".txt"]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple(map(int, i.split("\t")[0:3])) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def load_label(self, data_dir,file_num):

        file_names = [data_dir+".txt"]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple(map(int, i.split("\t")[0:file_num])) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple
    def load_dict(self, data_dir, file_num):
        if file_num == 4:
            file_names = [data_dir + str(i)+".txt" for i in range(1, 5)]
        elif file_num == 3:
            file_names = [data_dir + str(i)+".txt" for i in range(1, 4)]
        else:
            file_names = [data_dir]
        what2id, id2what, ids = {}, {}, []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data]
                what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
                id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
                ids.append(set([int(i[0]) for i in data]))
        return what2id, id2what, ids

    def gen_sparse_graph_from_triples(self, triples, ins_num, with_r=False):
        edge_dict = {}
        for (h, r, t) in triples:
            # if h != t:
            if (h, t) not in edge_dict:
                edge_dict[(h, t)] = []
                edge_dict[(t, h)] = []
            edge_dict[(h, t)].append(r)
            edge_dict[(t, h)].append(-r)
        if with_r:
            edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            r_ij = [int(abs(r)) for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            edges = np.array(edges, dtype=np.int32)
            values = np.array(values, dtype=np.float32)
            r_ij = np.array(r_ij, dtype=np.float32)
            return edges, values, r_ij
        else:
            edges = [[h, t] for (h, t) in edge_dict]
            values = [1 for (h, t) in edge_dict]
        # add self-loop
        edges += [[e, e] for e in range(ins_num)]
        values += [1 for e in range(ins_num)]
        edges = np.array(edges, dtype=np.int32)
        values = np.array(values, dtype=np.float32)
        return edges, values, None

    def share(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        new_triples = []
        for (h, r, t) in triples:
            if h in from_1_to_2_dict:
                h = from_1_to_2_dict[h]
            if t in from_1_to_2_dict:
                t = from_1_to_2_dict[t]
            new_triples.append((h, r, t))
        new_triples = list(set(new_triples))
        return new_triples

    def swap(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        from_2_to_1_dict = dict(ill[:, ::-1])
        new_triples = []
        for (h, r, t) in triples:
            new_triples.append((h, r, t))
            if h in from_1_to_2_dict:
                new_triples.append((from_1_to_2_dict[h], r, t))
            if t in from_1_to_2_dict:
                new_triples.append((h, r, from_1_to_2_dict[t]))
            if h in from_2_to_1_dict:
                new_triples.append((from_2_to_1_dict[h], r, t))
            if t in from_2_to_1_dict:
                new_triples.append((h, r, from_2_to_1_dict[t]))
        new_triples = list(set(new_triples))
        return new_triples

    def __repr__(self):
        return self.__class__.__name__ + " dataset summary:" + \
            "\n\tins_num: " + str(self.ins_num) + \
            "\n\trel_num: " + str(self.rel_num) + \
            "\n\ttriple_idx: " + str(len(self.triple_idx)) + \
            "\n\trate: " + str(self.rate) + "\tval: " + str(self.val) + \
            "\n\till_idx(train/test/val): " + str(len(self.ill_idx)) + " = " + str(
                len(self.ill_train_idx)) + " + " + str(len(self.ill_test_idx)) + " + " + str(len(self.ill_val_idx)) + \
            "\n\tins_G_edges_idx: " + str(len(self.ins_G_edges_idx)) + \
            "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"

    def _kg2hin_data(self):
        self.triple_idx = np.array(self.triple_idx)

        self.head = self.triple_idx[:, 0]
        self.rela = self.triple_idx[:, 1]
        self.tail = self.triple_idx[:, 2]
        values = torch.ones(self.triple_idx.shape[0])

        A_ht = sp.coo_matrix((values, (self.head, self.tail)), shape=(self.ins_num, self.ins_num))
        A_th = sp.coo_matrix((values, (self.tail, self.head)), shape=(self.ins_num, self.ins_num))
        A_hr = sp.coo_matrix((values, (self.head, self.rela)), shape=(self.ins_num, self.rel_num))
        A_rh = sp.coo_matrix((values, (self.rela, self.head)), shape=(self.rel_num, self.ins_num))
        A_tr = sp.coo_matrix((values, (self.tail, self.rela)), shape=(self.ins_num, self.rel_num))
        A_rt = sp.coo_matrix((values, (self.rela, self.tail)), shape=(self.rel_num, self.ins_num))

        self.adj_dict = {'H': {}, 'R': {}, 'T': {}}
        self.adj_dict['H']['T'] = _sparse_mx_to_torch_sparse_tensor(_row_normalize(A_ht)).to(self.device)
        self.adj_dict['H']['R'] = _sparse_mx_to_torch_sparse_tensor(_row_normalize(A_hr)).to(self.device)
        self.adj_dict['R']['H'] = _sparse_mx_to_torch_sparse_tensor(_row_normalize(A_rh)).to(self.device)
        self.adj_dict['R']['T'] = _sparse_mx_to_torch_sparse_tensor(_row_normalize(A_rt)).to(self.device)
        self.adj_dict['T']['H'] = _sparse_mx_to_torch_sparse_tensor(_row_normalize(A_th)).to(self.device)
        self.adj_dict['T']['R'] = _sparse_mx_to_torch_sparse_tensor(_row_normalize(A_tr)).to(self.device)



def _row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
if __name__ == '__main__':
    # TEST

    d = AlignmentData(share=False, swap=False)
    print(d)
    d = AlignmentData(share=True, swap=False)
    print(d)
    d = AlignmentData(share=False, swap=True)
    print(d)

import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp


class BaseLoader():
    
    def __init__(self, args):
        self.arg = args

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.kg_file = os.path.join(self.data_dir, "kg.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)

        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
    
    def load_cf(self, filename):
        """
        将原始数据转换为interactions
        """
        users, items = [], []
        user_dict = {}

        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                interactions = [int(t) for t in line.strip().split()]
                if (len(interactions) < 2):
                    continue
                user_id, item_ids = interactions[0], interactions[1:]
                item_ids = list(set(item_ids))
                for item_id in item_ids:
                    users.append(user_id)
                    items.append(item_id)
                user_dict[user_id] = item_ids
        
        users = np.array(users, dtype=np.int32)
        items = np.array(items, dtype=np.int32)
        return [users, items], user_dict

    def sample_pos_items_for_user(self, user_dict, user, n_items):
        user_items = user_dict[user]
        sample_items = []
        while len(sample_items) < n_items:
            item = random.choice(user_items)
            if item not in sample_items:
                sample_items.append(item)
        return sample_items
    
    def sample_neg_items_for_user(self, user_dict, user, n_items):
        user_items = user_dict[user]
        sample_items = []
        while len(sample_items) < n_items:
            item = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if item not in user_items and item not in sample_items:
                sample_items.append(item)
        return sample_items


    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = user_dict.keys()
        if batch_size <= len(exist_users):
            batch_users = random.sample(exist_users, batch_size)
        else:
            batch_users = [random.choice(exist_users) for _ in range(batch_size)]
        batch_pos_items, batch_neg_items = [], []
        for user in batch_users:
            batch_pos_items += self.sample_pos_items_for_user(user_dict, user, 1)
            batch_neg_items += self.sample_neg_items_for_user(user_dict, user, 1)
        
        batch_users = torch.LongTensor(batch_users)
        batch_pos_items = torch.LongTensor(batch_pos_items)
        batch_neg_items = torch.LongTensor(batch_neg_items)
        return batch_users, batch_pos_items, batch_neg_items


class DataLoader(BaseLoader):
    
    def __init__(self, args):
        super().__init__(args)

        self.kg_data = self.load_kg(self.kg_file)

        self.n_relations = max(self.kg_data['r']) + 1
        self.n_entities = max(max(self.kg_data['h']), max(self.kg_data['t'])) + 1

        self.train_data = self.get_train_data()

        self.n_relations = max(self.train_data['r']) + 1

        self.n_train_data = len(self.train_data)

        self.n_users_entities = self.n_users + self.n_entities

        self.h_list, self.r_list, self.t_list, self.train_h_dict, self.train_r_dict = self.get_train_dict()

        self.adjacency_dict = self.get_adjacency_dict()

        self.laplacian_dict = self.get_laplacian_dict()

        self.A_in = self.get_A_in()

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'])
        kg_data = kg_data.drop_duplicates()
        return kg_data
    
    def get_train_data(self):
        kg_data = self.kg_data

        # add inverse kg data
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += self.n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], ignore_index=True)

        # re-map user id
        def remap_user_id(data, n):
            data[0] = [t + n for t in data[0]]
            data[0] = np.array(data[0], dtype=np.int32)
            return data
        
        def remap_user_dict(data, n):
            data = {k + n: np.array(v, dtype=np.int32) for k, v in data.items()}
            return data

        self.cf_train_data = remap_user_id(self.cf_train_data, self.n_entities)
        self.cf_test_data = remap_user_id(self.cf_test_data, self.n_entities)

        self.train_user_dict = remap_user_dict(self.train_user_dict, self.n_entities)
        self.test_user_dict = remap_user_dict(self.test_user_dict, self.n_entities)

        # add cf to kg
        kg_data['r'] += 2

        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)

        return kg_train_data

    def get_train_dict(self):
        h_list, r_list, t_list = [], [], []
        train_h_dict, train_r_dict = defaultdict(list), defaultdict(list)

        for row in self.train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            r_list.append(r)
            t_list.append(t)
            train_h_dict[h].append([r, t])
            train_r_dict[r].append([h, t])
        
        h_list = torch.LongTensor(h_list)
        r_list = torch.LongTensor(r_list)
        t_list = torch.LongTensor(t_list)
        return h_list, r_list, t_list, train_h_dict, train_r_dict
    
    def get_adjacency_dict(self):
        adj_dict = {}
        for k, v in self.train_r_dict.items():
            rows = [t[0] for t in v]
            cols = [t[1] for t in v]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            adj_dict[k] = adj
        return adj_dict
    
    def get_laplacian_dict(self):
        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1), dtype=np.float).flatten()  # 按行加和
            d_inv = np.reciprocal(rowsum, where=rowsum != 0)  # 非零项取倒数
            d_mat_inv = sp.diags(d_inv)  # 生成对角矩阵
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()
        
        lap_dict = {}
        for k, v in self.adjacency_dict.items():
            lap_dict[k] = random_walk_norm_lap(v)
        return lap_dict
    
    def get_A_in(self):
        # 矩阵按元素相加 返回sparse tensor类型
        A_in = sum(self.laplacian_dict.values())
        A_in = A_in.tocoo()
        values = A_in.data
        indices = np.vstack((A_in.row, A_in.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = A_in.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    def sample_pos_triples_for_head(self, h_dict, head, n_items):
        pos_triples = h_dict[head]
        sample_triples, sample_relations, sample_tails = [], [], []
        while len(sample_relations) < n_items:
            triple = random.choice(pos_triples)
            if triple not in sample_triples:
                relation, tail = triple
                sample_triples.append(triple)
                sample_relations.append(relation)
                sample_tails.append(tail)
        return sample_relations, sample_tails
    
    def sample_neg_triples_for_head(self, h_dict, head, relation, n_items):
        pos_triples = h_dict[head]
        sample_tails = []
        while len(sample_tails) < n_items:
            tail = np.random.randint(low=0, high=self.n_users_entities, size=1)[0]
            if tail not in sample_tails and [relation, tail] not in pos_triples:
                sample_tails.append(tail)
        return sample_tails
    
    def generate_kg_batch(self, h_dict, batch_size, n_users_entities):
        exist_heads = h_dict.keys()
        if batch_size <= len(exist_heads):
            batch_heads = random.sample(exist_heads, batch_size)
        else:
            batch_heads = [random.choice(exist_heads) for _ in range(batch_size)]
        
        batch_relations, batch_pos_tails, batch_neg_tails = [], [], []
        for head in batch_heads:
            relations, pos_tails = self.sample_pos_triples_for_head(h_dict, head, 1)
            batch_relations += relations
            batch_pos_tails += pos_tails

            neg_tails = self.sample_neg_triples_for_head(h_dict, head, relations[0], 1)
            batch_neg_tails += neg_tails
        
        batch_heads = torch.LongTensor(batch_heads)
        batch_relations = torch.LongTensor(batch_relations)
        batch_pos_tails = torch.LongTensor(batch_pos_tails)
        batch_neg_tails = torch.LongTensor(batch_neg_tails)
        return batch_heads, batch_relations, batch_pos_tails, batch_neg_tails

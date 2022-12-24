import numpy as np
import torch


def precision_at_k_batch(hits, k):
    res = hits[:, :k].mean(axis=1)
    return res


def recall_at_k_batch(hits, k):
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def ndcg_at_k_batch(hits, k):
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, users, items, Ks):
    test_pos_item_binary = np.zeros([len(users), len(items)], dtype=np.float32)

    for idx, user in enumerate(users):
        train_pos_item_list = train_user_dict[user]
        test_pos_item_list = test_user_dict[user]
        cf_scores[idx][train_pos_item_list] = -np.inf  # 去掉已经推荐过的
        test_pos_item_binary[idx][test_pos_item_list] = 1
    
    _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(users)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)
    
    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall']    = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg']      = ndcg_at_k_batch(binary_hit, k)
    return metrics_dict
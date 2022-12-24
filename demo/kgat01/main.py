import os, sys
import random
import time
import numpy as np
import torch
import torch.optim as optim
import logging
from tqdm import tqdm

from parser import parse_args
from log_helper import *
from dataloader import *
from kgat import *
from metrics import *


def evaluate(args, data, model, Ks, device):
    model.eval()

    users = list(data.test_user_dict.keys())

    user_batches = [users[i: i + args.test_batch_size] for i in range(0, len(users), args.test_batch_size)]
    user_batches = [torch.LongTensor(d) for d in user_batches]

    items = torch.arange(data.n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_batches), desc='Evaluating Iteration') as pbar:
        for user_batch in user_batches:
            user_batch = user_batch.to(device)
            with torch.no_grad():
                batch_scores = model(user_batch, items, mode='predict')
            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, data.train_user_dict, data.test_user_dict, user_batch.cpu().numpy(), items.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)
    
    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log_{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data = DataLoader(args)

    model = KGAT(args, data)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # train
    for epoch in range(1, args.n_epoch + 1):
        time0 = time.time()
        model.train()
        
        # train cf
        # time1 = time.time()
        # cf_total_loss = 0
        # n_cf_batch = data.n_cf_train // args.cf_batch_size + 1
        # for iter in range(1, n_cf_batch + 1):
        #     time2 = time.time()
        #     cf_batch_users, cf_batch_pos_items, cf_batch_neg_items = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)

        #     cf_batch_users = cf_batch_users.to(device)
        #     cf_batch_pos_items = cf_batch_pos_items.to(device)
        #     cf_batch_neg_items = cf_batch_neg_items.to(device)
            
        #     cf_batch_loss = model(cf_batch_users, cf_batch_pos_items, cf_batch_neg_items, mode='train_cf')
        #     cf_batch_loss.backward()

        #     cf_optimizer.zero_grad()
        #     cf_optimizer.step()

        #     cf_total_loss += cf_batch_loss.item()
            
        #     logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time.time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        # logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time.time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        # time3 = time.time()
        # kg_total_loss = 0
        # n_kg_batch = data.n_train_data // args.kg_batch_size + 1
        # for iter in range(1, n_kg_batch + 1):
        #     time4 = time.time()

        #     kg_batch_heads, kg_batch_relations, kg_batch_pos_tails, kg_batch_neg_tails = data.generate_kg_batch(data.train_h_dict, args.kg_batch_size, data.n_users_entities)
        #     kg_batch_heads = kg_batch_heads.to(device)
        #     kg_batch_relations = kg_batch_relations.to(device)
        #     kg_batch_pos_tails = kg_batch_pos_tails.to(device)
        #     kg_batch_neg_tails = kg_batch_neg_tails.to(device)

        #     kg_batch_loss = model(kg_batch_heads, kg_batch_relations, kg_batch_pos_tails, kg_batch_neg_tails, mode='train_kg')

        #     kg_batch_loss.backward()

        #     kg_optimizer.zero_grad()
        #     kg_optimizer.step()

        #     kg_total_loss += kg_batch_loss.item()

        #     logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time.time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        # logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time.time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        # time5 = time.time()
        # h_list = data.h_list.to(device)
        # r_list = data.r_list.to(device)
        # t_list = data.t_list.to(device)
        # relations = list(data.laplacian_dict.keys())
        # model(h_list, t_list, r_list, relations, mode='update_att')
        # logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time.time() - time5))

        # logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time.time() - time0))

        # evalate cf
        time6 = time.time()
        _, metrics_dict = evaluate(args, data, model, Ks, device)
        logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time.time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
        
        epoch_list.append(epoch)
        for k in Ks:
            for m in ['precision', 'recall', 'ndcg']:
                metrics_list[k][m].append(metrics_dict[k][m])
        best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

        if should_stop:
                break

        if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
            save_model(model, args.save_dir, epoch, best_epoch)
            logging.info('Save model on epoch {:04d}!'.format(epoch))
            best_epoch = epoch



if __name__ == "__main__":
    os.chdir(sys.path[0])
    args = parse_args()
    train(args)
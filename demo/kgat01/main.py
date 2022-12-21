import os, sys
import random
import time
import numpy as np
import torch
import torch.optim as optim
import logging

from parser import parse_args
from log_helper import *
from dataloader import *
from kgat import *


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
        time1 = time.time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // args.cf_batch_size + 1
        for iter in range(1, n_cf_batch + 1):
            time2 = time.time()
            cf_batch_users, cf_batch_pos_items, cf_batch_neg_items = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)

            cf_batch_users = cf_batch_users.to(device)
            cf_batch_pos_items = cf_batch_pos_items.to(device)
            cf_batch_neg_items = cf_batch_neg_items.to(device)
            
            cf_batch_loss = model(cf_batch_users, cf_batch_pos_items, cf_batch_neg_items, mode='train_cf')
            cf_batch_loss.backward()

            cf_optimizer.zero_grad()
            cf_optimizer.step()

            cf_total_loss += cf_batch_loss.item()
            
            logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time.time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time.time() - time1, cf_total_loss / n_cf_batch))




if __name__ == "__main__":
    print('current path', os.getcwd())
    print('sys path:', sys.path[0])
    os.chdir(sys.path[0])
    args = parse_args()
    train(args)
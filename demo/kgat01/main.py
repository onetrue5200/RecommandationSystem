import random
import numpy as np
import torch
import logging

from parser import parse_args
from log_helper import *
from dataloader import *


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


if __name__ == "__main__":
    args = parse_args()
    train(args)
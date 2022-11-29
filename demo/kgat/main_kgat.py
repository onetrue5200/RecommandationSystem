import os
import random

import numpy as np

import torch

from parser.parser_kgat import *
from utils.log_helper import *
from data_loader.loader_kgat import *


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log_{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU/CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)


if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)

import random
import numpy as np
import torch

from parser import parse_args
from log_helper import *


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
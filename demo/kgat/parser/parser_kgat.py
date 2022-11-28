import argparse


def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT")

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')

    args = parser.parse_args()

    print(args)

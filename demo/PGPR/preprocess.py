import os, sys

import argparse
import pickle

from dataset import Dataset
from knowledge_graph import KnowledgeGraph


def generate_labels(mode='train'):
    review_file = '../../datasets/Amazon_Beauty/{}.txt'.format(mode)
    user_products = {}  # {uid: [pid,...], ...}
    with open(review_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(user_products, mode=mode)


def save_labels(labels, mode='train'):
    path = "../../tmp/Amazon_Beauty/"
    if mode == 'train':
        filename = 'train_label.pkl'
    elif mode == 'test':
        filename = 'test_label.pkl'
    else:
        raise Exception('mode should be one of {train, test}.')
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + filename, 'wb') as f:
        pickle.dump(labels, f)


def save_dataset(dataset_obj):
    dataset_path = '../../tmp/Amazon_Beauty/'
    dataset_filename = 'dataset.pkl'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    with open(dataset_path + dataset_filename, 'wb') as f:
        pickle.dump(dataset_obj, f)


def save_kg(kg):
    path = '../../tmp/Amazon_Beauty/'
    filename = 'kg.pkl'
    pickle.dump(kg, open(path + filename, 'wb'))


def load_dataset():
    dataset_path = '../../tmp/Amazon_Beauty/'
    dataset_filename = 'dataset.pkl'
    dataset_obj = pickle.load(open(dataset_path + dataset_filename, 'rb'))
    return dataset_obj


def main():
    print("preprocessing dataset")
    # load dataset
    dataset = Dataset('../../datasets/Amazon_Beauty')
    save_dataset(dataset)

    # generate KG
    # dataset = load_dataset()
    # kg = KnowledgeGraph(dataset)
    # kg.compute_degrees()
    # save_kg(kg)

    # generate labels
    # print('Generate train/test labels.')
    # generate_labels('train')
    # generate_labels('test')

if __name__ == '__main__':
    os.chdir(sys.path[0])
    main()
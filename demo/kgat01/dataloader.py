import os
import numpy as np

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
        return (users, items), user_dict

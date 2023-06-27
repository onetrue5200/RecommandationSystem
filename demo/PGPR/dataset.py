import numpy as np
from easydict import EasyDict as edict

class Dataset():
    
    def __init__(self, data_dir, word_sampling_rate=1e-4) -> None:
        self.data_dir = data_dir + '/'
        self.review_file = self.data_dir + "train.txt"
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

    def _load_file(self, filename):
        with open(self.data_dir + filename, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def load_entities(self):
        entity_file = edict(
                user='users.txt',
                product='product.txt',
                word='vocab.txt',
                related_product='related_product.txt',
                brand='brand.txt',
                category='category.txt',
        )
        for name in entity_file:
            vocab = self._load_file(entity_file[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print('Load Entities', name, 'of size', len(vocab))

    def load_product_relations(self):
        product_relations = edict(
                produced_by=('brand_p_b.txt', self.brand),
                belongs_to=('category_p_c.txt', self.category),
                also_bought=('also_bought_p_p.txt', self.related_product),
                also_viewed=('also_viewed_p_p.txt', self.related_product),
                bought_together=('bought_together_p_p.txt', self.related_product),
        )
        for name in product_relations:
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) # tail's distrib
            )
            for line in self._load_file(product_relations[name][0]):
                knowledge = []  # edge tails
                for x in line.split(' '):
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1  # record tail's number
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load Relations', name, 'of size', len(relation.data))

    def load_reviews(self):
        review_data = []
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
            review_data.append((user_idx, product_idx, word_indices))
            product_distrib[product_idx] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)
        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                word_distrib=word_distrib,
                word_count=word_count,
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'word count =', word_count)

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)
            

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer


class KnowledgeGraph():
    
    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_reviews(dataset)
        self._load_knowledge(dataset)
        self._clean()
        self.top_matches = None
    
    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities():
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: [] for r in get_relations(entity)}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))
    
    def _load_reviews(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):
        print('Load reviews...')
        # (1) Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        distrib = dataset.review.word_distrib

        num_purchase_edges = 0
        num_mention_edges = 0
        all_removed_words = []
        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(review).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            if len(remained_words) <= 0:
                continue

            # (2) Add edges.
            self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)
            num_purchase_edges += 2
            for wid in remained_words:
                self._add_edge(USER, uid, MENTION, WORD, wid)
                self._add_edge(PRODUCT, pid, DESCRIBED_AS, WORD, wid)
                num_mention_edges += 4
        print('Total purchase {:d} review edges.'.format(num_purchase_edges))
        print('Total mention {:d} review edges.'.format(num_mention_edges))

        with open('../../tmp/review_removed_words.txt', 'w') as f:
            f.writelines([' '.join(words) + '\n' for words in all_removed_words])
    
    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)
    
    def _load_knowledge(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
    
    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data
    
    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

# Entities
USER = 'user'
PRODUCT = 'product'
WORD = 'word'
RPRODUCT = 'related_product'
BRAND = 'brand'
CATEGORY = 'category'
# Relations
PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
PRODUCED_BY = 'produced_by'
BELONG_TO = 'belongs_to'
ALSO_BOUGHT = 'also_bought'
ALSO_VIEWED = 'also_viewed'
BOUGHT_TOGETHER = 'bought_together'
SELF_LOOP = 'self_loop'  # only for kg env
KG_RELATION = {
    USER: {
        PURCHASE: PRODUCT,
        MENTION: WORD,
    },
    WORD: {
        MENTION: USER,
        DESCRIBED_AS: PRODUCT,
    },
    PRODUCT: {
        PURCHASE: USER,
        DESCRIBED_AS: WORD,
        PRODUCED_BY: BRAND,
        BELONG_TO: CATEGORY,
        ALSO_BOUGHT: RPRODUCT,
        ALSO_VIEWED: RPRODUCT,
        BOUGHT_TOGETHER: RPRODUCT,
    },
    BRAND: {
        PRODUCED_BY: PRODUCT,
    },
    CATEGORY: {
        BELONG_TO: PRODUCT,
    },
    RPRODUCT: {
        ALSO_BOUGHT: PRODUCT,
        ALSO_VIEWED: PRODUCT,
        BOUGHT_TOGETHER: PRODUCT,
    }
}

def get_entities():
    return list(KG_RELATION.keys())

def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())

def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]

def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf
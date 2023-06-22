import os
import logging
from collections import Counter
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'
PUNCTS = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]


class Config(object):
    with_punct = True
    unlabeled = False
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel = [self.root_label] + sorted(list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label])))
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        config = Config()
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep

        if self.unlabeled:
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # logging.info('Build dictionary for part-of-speech tags.')
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # logging.info('Build dictionary for words.')
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        self.n_tokens = len(tok2id)
        self.model = None

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        ### TODO:
        ###     You should implement your feature extraction here.
        ###     Extract the features for one example, ex
        ###     The features could include the word itself, the part-of-speech and so on.
        ###     Every feature could be represented by a string,
        ###     and the string can be converted to an id(int), according to the self.tok2id
        ###     Return: A list of token_ids corresponding to tok2id
        
        # print(stack, buf, arcs, ex)
        if stack[0] == 'ROOT':
            stack[0] = 0

        '''
        The paper says: 
        Sw contains nw = 18 elements: 
        (1) The top 3 words on the stack and buffer:
            s1, s2, s3, b1, b2, b3;
        (2) The first and second leftmost / rightmost children of the top two words on the stack:
            lc1(si), rc1(si), lc2(si), rc2(si), i = 1, 2. 
        (3) The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack:
            lc1(lc1(si)), rc1(rc1(si)), i = 1, 2.
        We use the corresponding POS tags for St (nt = 18), 
        and the corresponding arc labels of words excluding those 6 words on the stack / buffer for Sl (nl = 12). 
        '''

        # self.use_pos: default True
        # self.use_dep: = use_dep & not unlabeled, default False
        # Default feature size: 36
        
        sw = []     # word features
        st = []     # POS tag features
        sl = []     # arc label features

        def get_lc(x):
            # get leftmost children
            return sorted([arc[1] for arc in arcs if arc[0] == x and arc[1] < x])
        
        def get_rc(x):
            # get rightmost children
            return sorted([arc[1] for arc in arcs if arc[0] == x and arc[1] > x], reverse=True)
        
        sw += [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]  # top 3 words on the stack
        sw += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf))       # top 3 words on the buffer

        if self.use_pos:
            st += [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]] # top 3 stack POS
            st += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf))    # top 3 buffer POS
        
        for i in range(2):
            # not enough elements in stack, replace features with NULL
            if i >= len(stack):
                sw += [self.NULL] * 6
                if self.use_pos:
                    st += [self.P_NULL] * 6
                if self.use_dep:
                    sl += [self.L_NULL] * 6
            else:
                x = stack[-i-1]     # top word on the stack
                lc = get_lc(x)
                rc = get_rc(x)
                llc = get_lc(lc[0]) if len(lc) else []
                rrc = get_rc(rc[0]) if len(rc) else []

                tmp_features = []
                for lst, idx in zip([lc, rc, lc, rc, llc, rrc], [0, 0, 1, 1, 0, 0]):
                    tmp_features.append(ex['word'][lst[idx]] if len(lst) > idx else self.NULL)
                sw += tmp_features

                if self.use_pos:
                    tmp_features = []
                    for lst, idx in zip([lc, rc, lc, rc, llc, rrc], [0, 0, 1, 1, 0, 0]):
                        tmp_features.append(ex['pos'][lst[idx]] if len(lst) > idx else self.P_NULL)
                    st += tmp_features
                
                if self.use_dep:
                    tmp_features = []
                    for lst, idx in zip([lc, rc, lc, rc, llc, rrc], [0, 0, 1, 1, 0, 0]):
                        tmp_features.append(ex['label'][lst[idx]] if len(lst) > idx else self.L_NULL)
                    sl += tmp_features
        features = sw + st + sl
        return features

    def get_oracle(self, stack, buf, ex):
        # TODO: 根据当前状态，返回应该执行的操作编号（对应__init__中的trans），若无操作则返回None。
        
        # self.trans: ['L-xxx'] * self.n_deprel + ['R-xxx'] * self.n_deprel + ['S']
        # self.n_trans: len(self.trans)
        if len(stack) < 2:
            return self.n_trans - 1 if len(buf) else None
        
        x0, x1 = stack[-1], stack[-2]
        head0 = ex['head'][x0]
        head1 = ex['head'][x1]
        label0 = ex['label'][x0]
        label1 = ex['label'][x1]

        if self.unlabeled:
            # x0 -> x1 (not ROOT): LEFT ARC
            if (x1 > 0) and (head1 == x0):
                return 0
            # x1 -> x0 and x0 points to no words left in buf: RIGHT ARC
            if (x1 >= 0) and (head0 == x1) and (not len([x for x in buf if ex['head'][x] == x0])):
                return 1
            # otherwise: SHIFT or None, depends on length of buf
            return 2 if len(buf) else None
        else:
            if (x1 > 0) and (head1 == x0):
                return label1 if (0 <= label1 < self.n_deprel) else None 
            if (x1 >= 0) and (head0 == x1) and (not len([x for x in buf if ex['head'][x] == x0])):
                return label0 + self.n_deprel if (0 <= label0 < self.n_deprel) else None 
            return self.n_trans - 1 if len(buf) else None

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                # TODO: 根据gold_t，更新stack, arcs, buf

                # SHIFT
                if gold_t == self.n_trans - 1:
                    stack.append(buf.pop(0))
                # LEFT ARC
                elif gold_t < self.n_deprel:
                    dep = stack.pop(-2)
                    arcs.append((stack[-1], dep, gold_t))
                # RIGHT ARC
                else:
                    dep = stack.pop()
                    arcs.append((stack[-1], dep, gold_t))
            else:
                succ += 1
                all_instances += instances

        return all_instances

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels

    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies, relations = minibatch_parse(sentences, model, eval_batch_size)

        # UAS = LAS = all_tokens = 0.0
        # with tqdm(total=len(dataset)) as prog:
        #     for i, ex in enumerate(dataset):
        #         head = [-1] * len(ex['word'])
        #         for h, t, in dependencies[i]:
        #             head[t] = h
        #         for pred_h, gold_h, pred_l, gold_l, pos in \
        #                 zip(head[1:], ex['head'][1:], relations[i], ex['label'][1:], ex['pos'][1:]):
        #                 assert self.id2tok[pos].startswith(P_PREFIX)
        #                 pos_str = self.id2tok[pos][len(P_PREFIX):]
        #                 if (self.with_punct) or (not (pos_str in PUNCTS)):
        #                     UAS += 1 if pred_h == gold_h else 0
        #                     LAS += 1 if ((pred_h == gold_h) and (pred_l % self.n_deprel == gold_l)) else 0
        #                     all_tokens += 1
        #         prog.update(i + 1)
        # UAS /= all_tokens
        # LAS /= all_tokens
        ##### The method of calculating LAS above is WRONG!! A modified version is implemented in utils.py

        return dependencies, relations


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_x = torch.from_numpy(mb_x).long()
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]

        pred = self.parser.model(mb_x)
        pred = pred.detach().numpy()
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        # pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
        action_pred = ["S" if p == self.parser.n_trans - 1 else ("LA" if 0 <= p < self.parser.n_deprel else "RA") for p in pred]
        return pred, action_pred


def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)
    ls = sorted(ls)
    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def load_and_preprocess_data(reduced=True):
    config = Config()

    print("Loading data...",)
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]

    print("Building parser...",)
    parser = Parser(train_set)

    print("Vectorizing data...",)
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)

    print("Preprocessing training data...",)
    train_examples = parser.create_instances(train_set)

    print("Preparing embeddings...")
    word_vec = {}
    for line in open('./data/newglove.6B.100d.txt', encoding='utf-8').readlines():
        sp = line.strip().split()
        word_vec[sp[0]] = [float(x) for x in sp[1:]]
    embeddings = np.random.normal(size=(parser.n_tokens, 100))

    for tok in parser.tok2id:
        i = parser.tok2id[tok]
        if tok in word_vec:
            embeddings[i] = word_vec[tok]
        elif tok.lower() in word_vec:
            embeddings[i] = word_vec[tok.lower()]

    return parser, train_examples, dev_set, test_set, embeddings
# Most of the codes are borrowed from 
# https://github.com/elikip/bist-parser/blob/master/barchybrid

import re, os
import time
import logging
import random
import torch
import numpy as np
from collections import Counter


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, 
                 feats=None, parent_id=None, relation=None,
                 deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.pos = pos
        self.cpos = cpos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.cpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, 
                  self.pred_relation, self.deps, self.misc]

        # values = [str(self.id), self.form, self.lemma, self.pos, self.cpos, self.feats,
        #           str(self.parent_id), self.relation, self.deps, self.misc]

        return '\t'.join(['_' if v is None else v for v in values])


class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = 0  # None
            root.pred_relation = 'rroot'  # None
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)

    def attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in range(len(sentence)):
        for i in range(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id] -= 1
                forest.attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id] -= 1
                forest.attach(i, i+1)
                break
    
    # Clear values changed by attach()
    for entry in sentence:
        entry.pred_parent_id = 0  # None

    return len(forest.roots) == 1



def read_conll(conllFP, proj=True):
    dropped = 0
    read = 0
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '-', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in conllFP:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1:
                if not proj or isProj([t for t in tokens if isinstance(t, ConllEntry)]):
                    yield tokens
                else:
                    # print('Non-projective sentence dropped')
                    dropped += 1
                read += 1
            tokens = [root]
        else:
            if line[0] == '#' or '=' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tok[0] = int(tok[0])
                tok[6] = int(tok[6]) if tok[6] != '_' else -1
                tokens.append(ConllEntry(*tok))

    if len(tokens) > 1:
        yield tokens

    # print(read, 'sentences read')
    # print(dropped, 'dropped non-projective sentences.')


def vocab(conll_path):
    wordCount = Counter()
    posCount = Counter()
    relCount = Counter()
    lemmaCount = Counter()
    cposCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordCount.update(
                [node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update(
                [node.pos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update(
                [node.relation for node in sentence if isinstance(node, ConllEntry)])
            lemmaCount.update(
                [node.lemma for node in sentence if isinstance(node, ConllEntry)])
            cposCount.update(
                [node.cpos for node in sentence if isinstance(node, ConllEntry)])

    print('the amount of kind of words, pos-tag, relations, ontology, cpos_tag:',
          len(wordCount), len(posCount), len(relCount), len(lemmaCount), len(cposCount))

    # construct vocabs
    # for simplicify, onto and cpos are not used here
    vocab = {'*UNK*': 0, '*PAD*': 1, '*INITIAL*': 2}
    word_vocab = {word: i+len(vocab) for i, word in enumerate(wordCount.keys())}
    word_vocab.update(vocab)
    pos_vocab = {pos: i+len(vocab) for i, pos in enumerate(posCount.keys())}
    pos_vocab.update(vocab)
    rel_vocab = {rel: i for i, rel in enumerate(relCount.keys())}

    return wordCount, word_vocab, pos_vocab, rel_vocab


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


def cal_uas_las(gold_fn, pred_fn):
    result_fn = pred_fn + '.txt'
    uas, las = 0., 0.
    
    if gold_fn.endswith('conll'):
        cmd = 'perl ../utils/eval.pl -g ' + gold_fn + ' -s ' + pred_fn + ' > ' + result_fn
    else:
        cmd = 'python3 ../utils/evaluation_script/conll17_ud_eval.py -v -w {0} {1} {2} > {3}'.format(
                        '../utils/evaluation_script/weights.clas',
                        gold_fn,
                        pred_fn,
                        result_fn)
    os.system(cmd)

    if gold_fn.endswith('conll'):
        with open(result_fn, 'r') as f:
            for l in f:
                if l.strip().startswith('Unlabeled attachment score'):
                    uas = float(l.strip().split()[-2])
                elif l.strip().startswith('Labeled   attachment score'):
                    las = float(l.strip().split()[-2])
    else:
        with open(result_fn, 'r') as f:
            for l in f:
                if l.startswith('UAS'):
                    uas = float(l.strip().split()[-1])
                elif l.startswith('LAS'):
                    las = float(l.strip().split()[-1])
    
    return uas, las


def get_logger(log_dir, dataset):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    pathname = "{}/{}_{}.txt".format(log_dir, dataset, time.strftime("%m-%d_%H-%M-%S"))
    # pathname = "{}/{}.txt".format(log_dir, dataset)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
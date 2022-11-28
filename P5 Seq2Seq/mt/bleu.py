# by wucx
# 2022-11-18

import math
import numpy as np
from collections import Counter


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    # 1-gram, 2-gram, 3-gram, 4-gram
    for n in range(1, 5):
        h_ngrams = Counter([tuple(hypothesis[i: i+n]) for i in range(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i: i+n]) for i in range(len(reference) + 1 - n)])
        
        stats.append(max([sum((h_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics"""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0

    (h, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x) / y) for x , y in zip(stats[2::2], stats[3::2])]) / 4.
    return math.exp(min([0, 1 - float(r) / h]) + log_bleu_prec)


def get_bleu(hypotheses, references):
    """ Get validation BLEU score for dev set,
        Only one reference is given.
    """
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, references):
        stats += np.array(bleu_stats(hyp, ref))
    return bleu(stats)

'''
Created on Oct 14, 2016

@author: wcx
'''

import os, sys
import argparse
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity 


def load_txt_embeddings(fn):
    '''each line: WORD dim_1 dim_2, ..., dim_n'''
    
    vocab = {}
    embeddings = []
    with open(fn, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            items = line.strip().split()
            word  = items[0]
            embed = [float(x) for x in items[1:]]
            vocab[word] = idx
            embeddings.append(embed)
            idx += 1

    embeddings = np.asarray(embeddings)

    return vocab, embeddings


def cal_distance(opts, w, voc, We):
    topn = opts.topn
    dostand = opts.dostand
    
    idx_word = dict([i, w] for w, i in voc.items())
    if dostand:
        We = preprocessing.scale(We)
    
    # calculate
    distances = np.asarray(cosine_similarity([We[voc[w]]], We)[0])
    sort_idx = np.argsort(distances)[: : -1]
    top_distances = distances[sort_idx[1 : topn]]
    top_words = [idx_word[i] for i in sort_idx[1 : topn]]
    
    return top_words, top_distances


def main(conf):
    print("Enter a word, then show the nearest words according to the cosine distance.")
    print("Enter two words, then show the distance between these two words.")
    print("\nEnter 'xx' to exit.")
    
    # load word embeddings
    voc, We = load_txt_embeddings(conf.embed)
    
    word = ""
    while(True):
        word = input("\nEnter a word/two words:")
        if word == "xx":
            print("Exit Program.")
            break
        else:
            words = word.strip().split()
            if len(words) == 0 or len(words) > 2:
                continue
            
            if words[0] not in voc or words[-1] not in voc:
                print("??? the wrong word: {0}".format(words))
                continue
            
            if len(words) == 1:
                word = words[0]
                top_words, distances = cal_distance(conf, word, voc, We)
                print("\n {0:>20}       {1}".format('[word]', '[distance]'))
                for w, d in zip(top_words, distances):
                    print("{0:>20}       {1:.10f}".format(w, d))
            if len(words) == 2:
                word1, word2 = words[0], words[1]
                distance = np.asarray(cosine_similarity([We[voc[word1]]], [We[voc[word2]]])[0])
                print("\n {0:>10}  {1:>10}  {2}".format(word1, word2, distance))
    return None

if __name__ == "__main__":
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser(description = "Calculate distances between two words.")
    
    desc = "file name of word embeddings"
    parser.add_argument('--embed', type = str, default = 'DSWE/embeddings/ltw_xin.embed.txt', help = desc)
    
    desc = 'show top n similiar words'
    parser.add_argument('--topn', type = int, default = 15, help = desc)
    
    desc = 'whether standrize word embeddings'
    parser.add_argument('--dostand', type = int, default = 0, help = desc)
    
    conf = parser.parse_args()
    main(conf)

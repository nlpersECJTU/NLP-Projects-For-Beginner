# Some codes are borrowed from 
# https://github.com/elikip/bist-parser/blob/master/barchybrid


import os, sys
import pickle
import os.path
import time
import random
import torch
import argparse

import utils
import torch.autograd
from torch import optim
from transitionmodel import TranModel
from config import Config
from itertools import chain
from operator import itemgetter



class Parser(object):
    def __init__(self, model, conf):
        super(Parser, self).__init__()
        self.model = model
        self.conf = conf
        self.oracle = conf.oracle
        self.optim = optim.Adam(model.parameters())


    def train(self, data):
        self.model.train()

        total_loss = 0
        e_acc, e_total = 0, 0
        r_acc, r_total = 0, 0

        ninf = -float('inf')
        start = time.time()
        
        for j, sentence in enumerate(data):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            self.model.forward(conll_sentence)
            
            # init stack and buf
            stack = utils.ParseForest([])
            buf = utils.ParseForest(conll_sentence[1:] + [conll_sentence[0]])

            loss_errs = []
            while not (len(buf) == 1 and len(stack) == 0):
                scores = self.model.cal_action_scores(stack, buf, True)
                scores.append([(None, 3, ninf, None)])

                # split stack as s0, s1 and alpha (the rest items)
                alpha = stack.roots[:-2] if len(stack) > 2 else []
                s1 = [stack.roots[-2]] if len(stack) > 1 else []
                s0 = [stack.roots[-1]] if len(stack) > 0 else []

                # split buffer as b(b0) and beta (the rest items)
                beta = buf.roots[1:] if len(buf) > 1 else []
                b = [buf.roots[0]] if len(buf) > 0 else []

                # see Section 4.1 (Kiperwasser and Goldberg 2016) for more details
                # Dynamic oracle, see (Goldberg and Nivre 2013) for more details
                left_cost  = (len([h for h in s1 + beta if h.id == s0[0].parent_id]) + 
                              len([d for d in b + beta if d.parent_id == s0[0].id])) if len(scores[0]) > 0 else 1
                right_cost = (len([h for h in b + beta if h.id == s0[0].parent_id]) + 
                              len([d for d in b + beta if d.parent_id == s0[0].id])) if len(scores[1]) > 0 else 1
                shift_cost = (len([h for h in s1 + alpha if h.id == b[0].parent_id]) + 
                              len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id])) if len(scores[2]) > 0 else 1
                costs = (left_cost, right_cost, shift_cost, 1)

                # the correct transition: 
                # correct left/right action with correct relation, or correct shift action 
                bestValid = max(( s for s in chain(*scores) 
                                  if costs[s[1]] == 0 and ( s[1] == 2 or  s[0] == stack.roots[-1].relation )), 
                                  key=itemgetter(2))
                bestWrong = max(( s for s in chain(*scores) 
                                  if costs[s[1]] != 0 or  ( s[1] != 2 and s[0] != stack.roots[-1].relation )), 
                                  key=itemgetter(2))
                                  
                # error-expoloration training, and aggressive exploration
                best = bestValid if ((not self.oracle) 
                                      or (bestValid[2] - bestWrong[2] > 1.0) 
                                      or (bestValid[2] > bestWrong[2] and random.random() > 0.1)) else bestWrong

                # conduct actions
                # s0, s1, b0
                # shift: move b0 from buffer to stack 
                if best[1] == 2:
                    stack.roots.append(buf.roots[0])
                    del buf.roots[0]
                
                # left arc: remove s0, add arc s0<--b0 with relation l
                elif best[1] == 0:
                    child = stack.roots.pop()
                    parent = buf.roots[0]
                    child.pred_parent_id = parent.id
                    child.pred_relation = best[0]

                # right arc: remove s0, add arc s1-->s0 with relation l 
                elif best[1] == 1:
                    child = stack.roots.pop()
                    parent = stack.roots[-1]
                    child.pred_parent_id = parent.id
                    child.pred_relation = best[0]
                
                # calculate loss
                if bestValid[2] < bestWrong[2] + 1.0:
                    loss_errs.append(bestWrong[3] - bestValid[3])
                    total_loss += 1.0 + bestWrong[2] - bestValid[2]
                
                # for calculating acc
                if best[1] != 2:
                    if child.pred_parent_id == child.parent_id:
                        e_acc +=1
                    if child.pred_relation == child.relation:
                        r_acc += 1

            # for calculating acc            
            e_total += len(conll_sentence)
            r_total += len(conll_sentence)

            # update parameters
            self.optim.zero_grad()
            if len(loss_errs) > 0:
                loss = torch.sum(torch.cat([_.unsqueeze(0) for _ in loss_errs]))
                loss.backward()
                self.optim.step()

            # show training information
            if (j + 1) % 100 == 0:
                conf.logg.info(
                      'Sent No:{0} '.format(j + 1) +
                      'Head Acc:{:.4f} '.format(e_acc / e_total) + 
                      'Rela Acc:{:.4f} '.format(r_acc / r_total) +
                      'Total Loss:{:.4f} '.format(total_loss / (j + 1)) +
                      'Time:{:.4f}'.format(time.time() - start)
                      )
                start = time.time()
                e_acc, e_total = 0, 0
                r_acc, r_total = 0, 0

            # for debug
            if conf.debug and j > 1000:
                break


    def eval(self, data, rela_vocab):
        self.model.eval()

        for _, sentence in enumerate(data):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            self.model.forward(conll_sentence)
            # 
            stack = utils.ParseForest([])
            buf = utils.ParseForest(conll_sentence[1:] + [conll_sentence[0]])

            while not (len(buf) == 1 and len(stack) == 0):
                scores = self.model.cal_action_scores(stack, buf, False)
                best = max(chain(*scores), key = itemgetter(2))                

                # conduct actions 
                if best[1] == 2:
                    stack.roots.append(buf.roots[0])
                    del buf.roots[0]
                
                elif best[1] == 0:
                    child = stack.roots.pop()
                    parent = buf.roots[0]
                    child.pred_parent_id = parent.id
                    child.pred_relation = best[0]

                elif best[1] == 1:
                    child = stack.roots.pop()
                    parent = stack.roots[-1]
                    child.pred_parent_id = parent.id
                    child.pred_relation = best[0]
                
            yield conll_sentence

    
    def save(self, fn):
        torch.save(self.model.state_dict(), fn)

    
    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    

if __name__ == '__main__':
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, default='config/en-ud-proj-conllu.json')
    # parser.add_argument('--config', type=str, default='config/zh-ud-proj-conllu.json')
    # parser.add_argument('--config', type=str, default='config/en-universal-conllu.json')
    # parser.add_argument('--config', type=str, default='config/en-universal-conll.json')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dtrain', type=str)
    parser.add_argument('--ddev', type=str)
    parser.add_argument('--dtest', type=str)
    
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--model', type=str)

    parser.add_argument('--wdims', type=int)
    parser.add_argument('--pdims', type=int)
    parser.add_argument('--rdims', type=int)

    parser.add_argument('--activation', type=str)
    parser.add_argument('--lstm_layers', type=int)
    parser.add_argument('--lstm_hidden_size', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--window', type=int)

    parser.add_argument('--oracle', type=int, help="1: true, 0: false")

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optim', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--debug', type=int, help="1: true, 0: false")


    args = parser.parse_args()
    conf = Config(args)
    logg = utils.get_logger(conf.logdir, conf.dataset)
    conf.logg = logg
    logg.info(conf)
    suffix = '.conllu' if conf.dtrain.endswith('conllu') else '.conll'
    # utils.set_seed(conf.seed)


    logg.info('Preparing vocabulary table')
    word_count, word_vocab, pos_vocab, rela_vocab = utils.vocab(conf.dtrain)
    with open(conf.vocab, 'wb') as fp:
        pickle.dump((word_count, word_vocab, pos_vocab, rela_vocab), fp)
    logg.info('Finished collection vocabulary')

    logg.info('Loading training data')
    with open(conf.dtrain) as f:
        dtrain = list(utils.read_conll(f))

    logg.info('Loading evaluation data')
    with open(conf.ddev) as f:
        ddev = list(utils.read_conll(f))

    logg.info('Initializing mst-parser:')
    model = TranModel(word_count, word_vocab, pos_vocab, rela_vocab, conf)
    parser = Parser(model, conf)
        
    best_uas, best_las = 0.0, 0.0
    for epoch in range(conf.epochs):
        # train
        random.shuffle(dtrain)
        logg.info('\nStarting epoch {0}'.format(epoch))
        parser.train(dtrain)

        # eval
        dev_pred_fn = os.path.join(conf.outdir, conf.dataset + '_dev_epoch_' + str(epoch) + suffix)
        utils.write_conll(dev_pred_fn, parser.eval(ddev, rela_vocab))
        uas, las = utils.cal_uas_las(conf.ddev, dev_pred_fn)
        mark = ''
        if uas + las > best_uas + best_las:
            best_uas, best_las = uas, las
            parser.save(conf.model)
            mark = '*'
        logg.info('Dev UAS: {0:.2f} {1}'.format(uas, mark))
        logg.info('Dev LAS: {0:.2f} {1}'.format(las, mark))

        # for debug
        if conf.debug:
            break
            

    # test after training
    logg.info('\nLoading test data')
    with open(conf.dtest) as f:
        dtest = list(utils.read_conll(f))
    parser.load(conf.model)
    test_pred_fn = os.path.join(conf.outdir, conf.dataset + '_test_pred' + suffix)
    utils.write_conll(test_pred_fn, parser.eval(dtest, rela_vocab))
    uas, las = utils.cal_uas_las(conf.dtest, test_pred_fn)
    logg.info('Test UAS: {0:.2f}'.format(uas))
    logg.info('Test LAS: {0:.2f}'.format(las))

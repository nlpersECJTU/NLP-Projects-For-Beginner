# Some codes are borrowed from 
# https://github.com/elikip/bist-parser/blob/master/bmstparser

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
from graphmodel import GraphModel
from config import Config



class Parser(object):
    def __init__(self, model, conf):
        super(Parser, self).__init__()
        self.model = model
        self.conf = conf
        self.optim = optim.Adam(model.parameters())


    def train(self, data):
        self.model.train()

        total_loss = 0
        e_acc, e_total = 0, 0
        r_acc, r_total = 0, 0
        start = time.time()
        for j, sentence in enumerate(data):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            gold_heads = [entry.parent_id for entry in conll_sentence]
            gold_relas = [rela_vocab[e.relation] for e in conll_sentence[1:]]
            
            head_scores, rela_scores, pred_heads = self.model.forward(conll_sentence, gold_heads)
            pred_relas = torch.argmax(rela_scores, dim=-1)
            _, pred_top2_relas = torch.topk(rela_scores, 2, dim=-1)

            # Hinge loss
            loss_heads = []
            for i, (p_head, g_head) in enumerate(zip(pred_heads, gold_heads)):
                if p_head != g_head:
                    loss_heads.append(head_scores[p_head][i] - head_scores[g_head][i])
            
            loss_relas = []
            for i, (p_relas, g_rela) in enumerate(zip(pred_top2_relas, gold_relas)):
                w_rela = p_relas[1] if p_relas[0] == g_rela else p_relas[0]
                if rela_scores[i][g_rela] - rela_scores[i][w_rela] < 1:
                    loss_relas.append(rela_scores[i][w_rela] - rela_scores[i][g_rela])
            
            # update
            self.optim.zero_grad()
            if len(loss_heads) > 0 or len(loss_relas) > 0:
                l_items = [_.unsqueeze(0) for _ in loss_heads + loss_relas]
                loss = torch.sum(torch.cat(l_items))    
                loss.backward()
                self.optim.step()
                total_loss += loss.cpu().item()

            # show
            e_acc += sum([1 for p, g in zip(pred_heads, gold_heads) if p == g])
            e_total += len(gold_heads)
            r_acc += sum([1 for p, g in zip(pred_relas, gold_relas) if p == g])
            r_total += len(gold_relas)
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
            if j > 1000:
                break


    def eval(self, data, rela_vocab):
        self.model.eval()
        idx_2_rela = {v:k for k, v in rela_vocab.items()}

        for _, sentence in enumerate(data):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            _, rela_scores, pred_heads = self.model.forward(conll_sentence)
            
            for entry, head in zip(conll_sentence, pred_heads):
                entry.pred_parent_id = head
            
            pred_relas = torch.argmax(rela_scores, dim=-1).numpy()
            conll_sentence[0].pred_relation = '_'
            for i, rela_idx in enumerate(pred_relas):
                conll_sentence[i+1].pred_relation = idx_2_rela[rela_idx]
            
            yield conll_sentence

    
    def save(self, fn):
        torch.save(self.model.state_dict(), fn)

    
    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    

if __name__ == '__main__':
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, default='config/en-ud-conllu.json')
    # parser.add_argument('--config', type=str, default='config/en-ud-proj-conllu.json')
    # parser.add_argument('--config', type=str, default='config/zh-ud-conllu.json')
    # parser.add_argument('--config', type=str, default='config/zh-ud-proj-conllu.json')
    # parser.add_argument('--config', type=str, default='config/en-universal-conllu.json')

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

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optim', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--seed', type=int)


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

    logg.info('Initializing mst-parser:')
    model = GraphModel(word_count, word_vocab, pos_vocab, rela_vocab, conf)
    parser = Parser(model, conf)

    logg.info('Loading training data')
    with open(conf.dtrain) as f:
        dtrain = list(utils.read_conll(f))

    logg.info('Loading evaluation data')
    with open(conf.ddev) as f:
        ddev = list(utils.read_conll(f))
        
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

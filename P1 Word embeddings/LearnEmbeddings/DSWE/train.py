import os, sys
import time
import torch
import argparse
import utils

import torch.nn as nn
import numpy as np
from model import AvgModel


def save_embeddings(model, vocab_idx_word, outdir):
    embeddings = model.state_dict()['embed.weight']
    embeddings = embeddings.cpu().detach().numpy()
    word_num = embeddings.shape[0]

    with open(outdir, 'w', encoding='utf-8') as f:
        for i in range(word_num):
            word = vocab_idx_word[i]
            embed = ' '.join(['{:.6f}'.format(_) for _ in embeddings[i]])
            print('{0} {1}'.format(word, embed), file=f)

    return


def train(conf, model, data, optimizer, criterion):
    model.train()

    start_time = time.time()
    for epoch in range(conf.epochs):
        batch_i = 0
        for batch in utils.batch_iter(data, conf.batch_size, True):
            arg1, arg2, y, len1, len2 = batch
            arg1 = torch.LongTensor(arg1).to(conf.device)
            arg2 = torch.LongTensor(arg2).to(conf.device)
            y    = torch.LongTensor(y).to(conf.device)
            # mask len  
            len1 = torch.LongTensor(len1)
            len2 = torch.LongTensor(len2)
            len1 = utils.get_mask(len1, conf.arg_max_len)
            len2 = utils.get_mask(len2, conf.arg_max_len)
            len1 = len1.to(conf.device)
            len2 = len2.to(conf.device)

            optimizer.zero_grad()
            predicts = model(arg1, arg2, len1, len2)
            loss = criterion(predicts, y)
            loss.backward()
            optimizer.step()

            batch_i += 1
            if batch_i % 10000 == 0:
                logg.info("")
                logg.info("Epoch: %3d, Batch: %5d, Train Loss: %.4f, Time: %.2f" % (
                           epoch, batch_i, loss, time.time() - start_time))
                start_time = time.time()

        if (epoch + 1) % 2 == 0:
            epoch_dir = conf.outdir + '.' + str(epoch + 1)
            logg.info("")
            logg.info("Save word embeddings into " + epoch_dir)
            save_embeddings(model, vocab_idx_word, epoch_dir)

    logg.info('Training Done')


if __name__ == '__main__':
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='ltw_xin')
    parser.add_argument('--dtrain', type=str, default='../corpus/ltw_xin.arg1.arg2.conn')
    # parser.add_argument('--dtrain', type=str, default='../corpus/debug.data')
    parser.add_argument('--outdir', type=str, default='embeddings')
    parser.add_argument('--logdir', type=str, default='log')

    parser.add_argument('--word_dim', type=int, default=100)
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--class_num', type=int, default=30)
    parser.add_argument('--arg_max_len', type=int, default=50)
    parser.add_argument('--word_cutoff', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)

    conf = parser.parse_args()
    # random seed
    # self.seed = 123
    conf.seed = np.random.randint(1e8)
    
    # logg path
    conf.logdir = "{}/{}.txt".format(conf.logdir, conf.dataset)
    # conf.logdir = "{}/{}_{}.txt".format(conf.logdir, conf.dataset, time.strftime("%m-%d_%H-%M-%S"))
    
    # embeddings path
    conf.outdir = "{}/{}.embed.txt".format(conf.outdir, conf.dataset)
    # conf.outdir = "{}/{}.embed_{}.txt".format(conf.outdir, conf.dataset, time.strftime("%m-%d_%H-%M-%S"))

    logg = utils.get_logger(conf.logdir)
    conf.logg = logg
    logg.info(conf)

    utils.set_seed(conf.seed)
    conf.device = torch.device('cuda:{0}'.format(conf.device) if torch.cuda.is_available() else 'cpu')

    # vocab and data
    logg.info('Loading data...')
    vocab, data = utils.load_data(conf, conf.dtrain)
    vocab_idx_word = {v : k for k, v in vocab.items()}
    conf.word_num = len(vocab)
    logg.info('Word number: %s' % conf.word_num)

    # build and train
    model = AvgModel(conf)
    model.to(conf.device)
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
    criterion = nn.CrossEntropyLoss()
    train(conf, model, data, optimizer, criterion)

    # after training, 
    # you can call distance.py to test the learned word embeddings

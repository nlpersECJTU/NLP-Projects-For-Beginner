import os, sys
import time
import torch
import argparse
import utils

import torch.nn as nn
import numpy as np
from model import HAN
from config import Config


def eval(model, data):
    model.eval()

    data_size = data[0].shape[0]
    acc = 0.
    rmse = 0.

    with torch.no_grad():
        for batch in utils.batch_iter(data, conf.batch_size, True):
            x, y, sen_lens, doc_lens = batch
            x = torch.LongTensor(x).to(conf.device)
            y = torch.LongTensor(y).to(conf.device)
            sen_lens = torch.LongTensor(sen_lens)
            doc_lens = torch.LongTensor(doc_lens)
            sen_lens = sen_lens.view(-1)
            sen_lens = utils.get_mask(sen_lens, conf.max_word_num)
            doc_lens = utils.get_mask(doc_lens, conf.max_sent_num)
            sen_lens = sen_lens.to(conf.device)
            doc_lens = doc_lens.to(conf.device)

            predicts = model(x, sen_lens, doc_lens)
            predicts = predicts.max(1)[1]
            predicts = predicts.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            acc += np.sum((predicts == y))
            rmse += np.sum((predicts - y) ** 2)

    acc = acc / data_size
    rmse = np.sqrt(rmse/data_size)
    return acc, rmse


def train(conf, train_data, dev_data, optimizer, criterion):
    train_size = train_data[0].shape[0]
    eval_at = conf.eval_at * train_size
    conf.eval_at = eval_at
    stop_at = conf.stop_at
    best_dev_acc = 0
    best_dev_rmse = 100

    start_time = time.time()
    for epoch in range(conf.epochs):
        if stop_at <= 0:
            break

        batch_i = 0
        for batch in utils.batch_iter(train_data, conf.batch_size, True):
            model.train()
            x, y, sen_lens, doc_lens = batch
            x = torch.LongTensor(x).to(conf.device)
            y = torch.LongTensor(y).to(conf.device)
            sen_lens = torch.LongTensor(sen_lens)
            doc_lens = torch.LongTensor(doc_lens)
            sen_lens = sen_lens.view(-1)
            sen_lens = utils.get_mask(sen_lens, conf.max_word_num)
            doc_lens = utils.get_mask(doc_lens, conf.max_sent_num)
            sen_lens = sen_lens.to(conf.device)
            doc_lens = doc_lens.to(conf.device)

            predicts = model(x, sen_lens, doc_lens)
            loss = criterion(predicts, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_i += 1

            eval_at -= x.size(0)
            if eval_at <= 0:
                improved = ''
                dev_acc, dev_rmse = eval(model, dev_data)
                
                if best_dev_acc <= dev_acc:
                    torch.save(model.state_dict(), conf.model)
                    best_dev_acc = dev_acc
                    best_dev_rmse = dev_rmse
                    stop_at = conf.stop_at
                    improved = '*'
                else:
                    stop_at -= 1
                
                end_time = time.time()
                logg.info("")
                logg.info("Epoch: %3d, Batch: %5d, Train Loss: %.4f, Dev Acc: %.2f, Dev RMSE: %.3f, Time: %.2f %s" % (
                    epoch, batch_i, loss, dev_acc * 100, dev_rmse, (end_time-start_time), improved))

                start_time = time.time()
                eval_at = conf.eval_at

    logg.info("Best dev acc: %.2f, Best dev rmse: %.3f" % (best_dev_acc * 100, best_dev_rmse))
    return


if __name__ == '__main__':
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, choices=['config/yelp2013.json', 'config/yelp2014.json', 'config/imdb.json'],
                        default='config/yelp2013.json')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dtrain', type=str)
    parser.add_argument('--ddev', type=str)
    parser.add_argument('--dtest', type=str)
    
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--logdir', type=str)

    parser.add_argument('--word_embed', type=str)
    parser.add_argument('--word_dim', type=int)
    parser.add_argument('--finetune', type=int)
    parser.add_argument('--word_hidden_size', type=int)
    parser.add_argument('--sent_hidden_size', type=int)
    
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--stop_at', type=int)
    parser.add_argument('--eval_at', type=int)
    
    # if no seed is specified, a random seed is used
    parser.add_argument('--seed', type=int)
    

    args = parser.parse_args()
    conf = Config(args)
    
    logg = utils.get_logger(conf.logdir)
    conf.logg = logg
    logg.info(conf)

    utils.set_seed(conf.seed)
    conf.device = torch.device('cuda:{0}'.format(conf.device) if torch.cuda.is_available() else 'cpu')
    
    # create vocab
    logg.info('Loading word embeddings')
    vocab, embeddings = utils.load_embedings(conf)
    embeddings = torch.Tensor(embeddings)
    
    # load training, dev and test data
    logg.info('Loading training data')
    train_data = utils.load_data(conf, vocab, 'train')
    logg.info('Loading dev data')
    dev_data = utils.load_data(conf, vocab, 'dev')
    logg.info('Loading test data')
    test_data = utils.load_data(conf, vocab, 'test')
    
    # build and train model
    model = HAN(conf, embeddings)
    model.to(conf.device)
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
    criterion = nn.CrossEntropyLoss() 
    train(conf, train_data, dev_data, optimizer, criterion)

    # load the model with best dev acc, then eval it on the test dataset 
    model.load_state_dict(torch.load(conf.model))
    test_acc, test_rmse = eval(model, test_data)
    logg.info("Test Acc: %.2f, Test RMSE: %.3f" % (test_acc * 100, test_rmse))
    logg.info("Done")

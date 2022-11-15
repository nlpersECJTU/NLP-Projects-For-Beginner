# by wucx, 2022/10/22

import os, sys
import time
import argparse

import torch
import torch.nn as nn
import numpy as np

import utils
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from layers.bert_ner import BertSoftmaxNER
from config import Config
from metrics import SeqEntityScore


def eval(model, data):
    model.eval()
    id2label = {i: label for i, label in enumerate(label_list)}
    metric = SeqEntityScore(id2label)

    eval_loss = 0.0
    eval_steps = 0
    with torch.no_grad():
        for batch in utils.batch_iter_tagging(data, conf.batch_size, False):
            x_batch, y_batch, mask_batch, segid_batch, len_batch = batch
            x_batch = torch.LongTensor(x_batch).to(conf.device)
            y_batch = torch.LongTensor(y_batch).to(conf.device)
            mask_batch = torch.LongTensor(mask_batch).to(conf.device)
            segid_batch = torch.LongTensor(segid_batch).to(conf.device)
            len_batch = torch.LongTensor(len_batch).to(conf.device)
            outputs = model(x_batch, mask_batch, segid_batch, y_batch)
            
            tmp_eval_loss, logits = outputs[: 2]
            eval_loss += tmp_eval_loss.item()
            eval_steps += 1

            preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            truths = y_batch.cpu().numpy().tolist()
            x_lens = len_batch.cpu().numpy().tolist()
            for i, lbl_seq in enumerate(truths):
                tmp1 = []
                tmp2 = []
                for j, m in enumerate(lbl_seq):
                    if j == 0:
                        continue
                    elif j == x_lens[i] - 1:
                        metric.update(label_path=tmp1, pred_path=tmp2)
                        break
                    else:
                        tmp1.append(truths[i][j])
                        tmp2.append(preds[i][j])

        eval_loss = eval_loss / eval_steps
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results["loss"] = eval_loss
    return results 


def train(conf, model, train_data, dev_data):
    train_size = train_data[0].shape[0]
    total_steps = conf.epochs * train_size
    warmup_steps = int(conf.warmup_proportion * total_steps)
    
    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    bert_params = list(model.bert.named_parameters())
    classifier_params = list(model.classifier.named_parameters())

    optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 
            'weight_decay': conf.weight_decay, 'lr': conf.bert_lr},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0, 'lr': conf.bert_lr},
            
            {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)], 
            'weight_decay': conf.weight_decay, 'lr': conf.other_lr},
            {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0, 'lr': conf.other_lr}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.other_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps)

    # Train!
    logg.info("**** Start training ****")
    best_results = {"f1":0.0, "acc":0.0, "recall": 0.0, "loss": 0.0}
    start_time = time.time()
    batch_i = 0
    for epoch in range(conf.epochs):
        for batch in utils.batch_iter_tagging(train_data, conf.batch_size, True):
            model.train()
            x_batch, y_batch, mask_batch, segid_batch, len_batch = batch
            x_batch = torch.LongTensor(x_batch).to(conf.device)
            y_batch = torch.LongTensor(y_batch).to(conf.device)
            mask_batch = torch.LongTensor(mask_batch).to(conf.device)
            segid_batch = torch.LongTensor(segid_batch).to(conf.device)
            len_batch = torch.LongTensor(len_batch).to(conf.device)
            outputs = model(x_batch, mask_batch, segid_batch, y_batch)

            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), conf.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            batch_i += 1

            if batch_i % conf.save_steps == 0:
                results = eval(model, dev_data)
                improved = ""

                if results["f1"] > best_results["f1"]:
                    torch.save(model.state_dict(), conf.ckpdir)
                    best_results = results
                    improved = " *"

                info = "Eval " + ", ".join([f'{key} : {value: .4f}' for key, value in results.items()])
                logg.info(info + improved)
                # logg.info(" Entity results: ")
                # for key in sorted(entity_info.keys()):
                #     info = f'{key}: ' + "-".join([f'{key}: {value}' for key, value in entity_info[key].items()])
                #     logg.info(info)

        logg.info("Epoch: %3d, Time: %.2f" % (epoch + 1, time.time()-start_time))
        start_time = time.time()

    info = "Best Eval " + ", ".join([f'{key} : {value: .4f}' for key, value in best_results.items()])
    logg.info(info)
    return


if __name__ == "__main__":
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, default='config/resume.softmax.json')
    
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dtrain', type=str)
    parser.add_argument('--ddev', type=str)
    parser.add_argument('--dtest', type=str)

    parser.add_argument('--bert_dir', type=str)
    parser.add_argument('--train_max_seq_length', type=str)
    parser.add_argument('--eval_max_seq_length', type=str)
    
    parser.add_argument('--ckpdir', type=str)
    parser.add_argument('--logdir', type=str)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    # if no seed is specified, a random seed is used
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    conf = Config(args)
    logg = utils.get_logger(conf.logdir)
    conf.logg = logg
    utils.set_seed(conf.seed)
    conf.device = torch.device('cuda:{0}'.format(conf.device) if torch.cuda.is_available() else 'cpu')

    # labels
    # label_list = utils.get_labels_bio(conf.dataset)
    label_list = utils.get_labels(conf.dataset, "bio")
    logg.info(label_list)
    conf.num_classes = len(label_list)
    logg.info(conf)
    
    # load training and dev data
    tokenizer = BertTokenizer.from_pretrained(conf.bert_dir)  # Bert tokenizer
    logg.info('Loading training data')
    train_data = utils.load_data_tagging(conf.dtrain, tokenizer, label_list, conf.train_max_seq_length)
    logg.info("Total {0} training instances".format(len(train_data[0])))
    logg.info('Loading dev data')
    dev_data = utils.load_data_tagging(conf.ddev, tokenizer, label_list, conf.eval_max_seq_length)
    logg.info("Total {0} dev instances".format(len(dev_data[0])))
    
    # build and train model
    model = BertSoftmaxNER(conf)
    model.to(conf.device)
    train(conf, model, train_data, dev_data)
    logg.info('Traning Done!')

    # test the trained model
    # sometimes, the test dataset (or the true label) is not available
    if conf.dataset not in ["cluener"]:
        logg.info('Loading test data')
        test_data = utils.load_data_tagging(conf.dtest, tokenizer, label_list, conf.eval_max_seq_length)
        logg.info("Total {0} test instances".format(len(test_data[0])))
        model.load_state_dict(torch.load(conf.ckpdir))
        results = eval(model, test_data)
        info = "Test " + ", ".join([f'{key} : {value: .4f}' for key, value in results.items()])
        logg.info(info)
        logg.info("")
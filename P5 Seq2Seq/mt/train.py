# by wucx, 2022/11/19

import os, sys
import time
import argparse
import torch
import torch.nn as nn
import utils

from torch import optim
from torch.optim import Adam
from layers.transformer import Transformer
from config import Config
from bleu import get_bleu
# from torchtext.data.metrics import bleu_score
from decode import batch_greedy_decode, beam_search, beam_search_less_gpu


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def train_epoch(model, data_iter, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_iter):
        src = batch.src
        tgt = batch.trg

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        # predcit the next word, which is the current label
        lbl = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
        optimizer.step()

        epoch_loss += loss.item()
        if i % 100 == 0:
            logg.info(f'step:{round((i / len(data_iter)) * 100, 2):.2f} %, loss: {loss.item():.4f}')

    return epoch_loss / len(data_iter)


def evaluate(model, data_iter, criterion, use_beam=False):
    model.eval()
    epoch_loss = 0
    candidates = []
    references = []
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            src = batch.src
            tgt = batch.trg
            
            # calculate loss based on the true previous words 
            # as that in the training phrase
            output = model(src, tgt[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            # predcit the next word, which is the current label
            lbl = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, lbl)
            epoch_loss += loss.item()

            # translate using 'greedy decode' or 'beam search'
            # the predicted previous words are used
            # calculate the bleu score based on the translation
            if use_beam:
                beam_search_method = beam_search_less_gpu if conf.less_gpu else beam_search
                batch_cand = beam_search_method(model, 
                                         src,
                                         conf.beam_size,
                                         conf.length_penalty, 
                                         conf.max_len,
                                         sos=conf.tgt_sos_idx,
                                         eos=conf.tgt_eos_idx)
            else:
                batch_cand = batch_greedy_decode(model, 
                                                 src, 
                                                 conf.decode_max_len, 
                                                 sos=conf.tgt_sos_idx,
                                                 eos=conf.tgt_eos_idx)
            
            batch_ref = []
            for j in range(batch.batch_size):
                tgt_words = utils.idx_to_word(tgt[j], loader.target.vocab)
                batch_cand[j] = utils.idx_to_word(batch_cand[j], loader.target.vocab)
                batch_ref.append(tgt_words)

            candidates.extend(batch_cand)
            references.extend(batch_ref)

            # for debug
            # if i > 2:
            #     break

    total_bleu = get_bleu(candidates, references)

    return epoch_loss / len(data_iter), total_bleu


if __name__ == "__main__":
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, default='config/multi30k_en_de.json')
    
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ckpdir', type=str)
    parser.add_argument('--logdir', type=str)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--d_model', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--ffn_hidden', type=int)
    parser.add_argument('--drop_prob', type=float)
    
    parser.add_argument('--init_lr', type=float)
    parser.add_argument('--factor', type=float)
    parser.add_argument('--adam_eps', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--clip', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epoch', type=int)

    parser.add_argument('--use_beam', type=int)   # 1 yes (use beam search for decode), 0 no (use greedy decode)  
    parser.add_argument('--decode_max_len', type=int)
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--length_penalty', type=float)
    parser.add_argument('--less_gpu', type=int)   # 1 yes, less gpu but more decoding time when beam search
    parser.add_argument('--seed', type=int)

    # args and logg
    args = parser.parse_args()
    conf = Config(args)
    logg = utils.get_logger(conf.logdir)
    conf.logg = logg
    utils.set_seed(conf.seed)
    conf.device = torch.device('cuda:{0}'.format(conf.device) if torch.cuda.is_available() else 'cpu')

    # load data
    # here, using torchtext to prepare data
    # we can also prepare data by tokenize, create_vocab, index data and batch data
    tokenizer = utils.Tokenizer()
    loader = utils.DataLoader(ext=('.en', '.de'),
                              tokenize_en=tokenizer.tokenize_en,
                              tokenize_de=tokenizer.tokenize_de,
                              init_token='<sos>',   # start of sentence
                              eos_token='<eos>'      # end of sentence
                              )
    train, valid, test = loader.make_dataset()
    loader.build_vocab(train, min_freq=2)
    train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test, 
                                                         batch_size=conf.batch_size,
                                                         device=conf.device)
    conf.src_pad_idx = loader.source.vocab.stoi['<pad>']
    conf.tgt_pad_idx = loader.target.vocab.stoi['<pad>']
    conf.tgt_sos_idx = loader.target.vocab.stoi['<sos>']
    conf.tgt_eos_idx = loader.target.vocab.stoi['<eos>']

    conf.enc_voc_size = len(loader.source.vocab)
    conf.dec_voc_size = len(loader.target.vocab)
    logg.info(conf)

    # construct model, optimizer and criterion
    model = Transformer(src_pad_idx=conf.src_pad_idx,
                        tgt_pad_idx=conf.tgt_pad_idx,
                        tgt_sos_idx=conf.tgt_sos_idx,
                        tgt_eos_idx=conf.tgt_eos_idx,
                        d_model=conf.d_model,
                        enc_voc_size=conf.enc_voc_size,
                        dec_voc_size=conf.dec_voc_size,
                        max_len=conf.max_len,
                        ffn_hidden=conf.ffn_hidden,
                        n_head=conf.n_heads,
                        n_layers=conf.n_layers,
                        drop_prob=conf.drop_prob,
                        device=conf.device                    
    ).to(conf.device)
    logg.info(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(init_weights)

    optimizer = Adam(params=model.parameters(),
                     lr = conf.init_lr,
                     weight_decay = conf.weight_decay,
                     eps = conf.adam_eps
                    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=conf.factor,
                                                     patience=conf.patience
                                                    )
    criterion = nn.CrossEntropyLoss(ignore_index=conf.tgt_pad_idx) 

    # training model
    best_loss = float('inf')
    best_bleu = 0.
    train_losses, valid_losses, bleus = [], [], []
    for step in range(conf.epoch):
        # for debug
        # if step > 0:
        #     print("debug ...")   # we can set breakpoint on this line

        start_time = time.time()
        train_loss = train_epoch(model, train_iter, optimizer, criterion)
        valid_loss, valid_bleu = evaluate(model, valid_iter, criterion, conf.use_beam)
        end_time = time.time()

        if step > conf.warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        bleus.append(valid_bleu)
        epoch_mins, epoch_secs = utils.cal_mins_secs(start_time, end_time)

        improved = ''
        # if valid_loss < best_loss:  # save the best model based on valid loss or bleu score?
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            best_loss = valid_loss
            torch.save(model.state_dict(), conf.ckpdir)
            improved = '*'
        
        logg.info(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s,' 
                  + f'Train Loss: {train_loss:.3f}, Val Loss: {valid_loss:.3f}, BLEU Score: {valid_bleu:.4f} {improved}')
            
    # test the trained model
    logg.info('Load the best checkpoint model ...')
    model.load_state_dict(torch.load(conf.ckpdir))
    test_loss, test_bleu = evaluate(model, test_iter, criterion, conf.use_beam)
    logg.info(f'Test Loss: {test_loss:.3f}, Test Score: {test_bleu:.4f}')
    logg.info("")

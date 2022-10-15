# Most of the code is borrowed from https://github.com/graykode/nlp-tutorial
# 10/12/2022, by wucx

import os, sys
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from tqdm import tqdm


def save_embeddings(outdir):
    embeddings = model.state_dict()['emb.weight']
    embeddings = embeddings.cpu().detach().numpy()
    word_num = embeddings.shape[0]

    with open(outdir, 'w', encoding='utf-8') as f:
        for i in range(word_num):
            word = vocab_idx_word[i]
            embed = ' '.join(['{:.6f}'.format(_) for _ in embeddings[i]])
            print('{0} {1}'.format(word, embed), file=f)


def batch_iter():
    # shuffle positions of center words
    shuffled_indices = np.random.permutation(np.arange(win_size, len(wid_sequence) - win_size))

    data_size = len(shuffled_indices)
    num_batches = int((data_size - 1) / batch_size) + 1

    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_size)

        inputs = []
        labels = []
        for center_pos in shuffled_indices[start_id: end_id]:
            left_pos = center_pos - win_size
            right_pos = center_pos + win_size + 1
            inputs.append(wid_sequence[left_pos : center_pos] + wid_sequence[center_pos + 1 : right_pos])
            labels.append(wid_sequence[center_pos])

        yield inputs, labels


# Model
class SimpCBOW(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        # WT can also regarded as word embeddings
        # or using the average of 'emb' and 'WT' as the final embeddings
        self.WT = nn.Linear(embedding_size, vocab_size, bias=False) 

    def forward(self, x):
        # x: [batch_size, win_size * 2]
        x_emb = self.emb(x)
        x_avg = torch.mean(x_emb, dim=1)
        o = self.WT(x_avg)
        return o


if __name__ == '__main__':
    os.chdir(sys.path[0])

    batch_size = 128
    embedding_size = 100
    win_size = 3   # context_size = 2 * win_size
    cut_off = 10
    outdir = 'embeddings/cbow.txt'

    # load data
    fn = '../corpus/ltw_xin.arg1.arg2.conn'
    # fn = '../corpus/debug.data'

    sentences = []
    with open(fn, 'r') as f:
        for line in tqdm(f):
            items = line.strip().split(' ||| ')
            sen = '<s> {0} {1} {2} </s>'.format(items[0], items[2], items[1])
            sentences.append(sen)

    # vocab
    print('Create vocab ...')
    word_sequence = ' '.join(sentences).split()
    counter = Counter(word_sequence)
    vocab = [ w for w, v in counter.items() if v > cut_off ]
    vocab = { w : idx for idx, w in enumerate(vocab)}
    vocab_idx_word = {v : k for k, v in vocab.items()}
    vocab_size = len(vocab)
    print('Vocab_size: %d' % vocab_size)

    # preprocess data, word -> idx
    print('Prepare data ...')
    wid_sequence = [vocab[w] for w in word_sequence if w in vocab]
    print('Len of word seq: %d' % len(wid_sequence))
    
    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SimpCBOW()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # training
    print('Training...')
    for epoch in range(10):
        start = time.time()
        batch_i = 0
        for batch in batch_iter():
            input_batch, target_batch = batch
            input_batch = torch.LongTensor(input_batch).to(device)
            target_batch = torch.LongTensor(target_batch).to(device)

            optimizer.zero_grad()
            predicts = model(input_batch)
            loss = criterion(predicts, target_batch)
            
            batch_i += 1
            if batch_i % 100000 == 0:
                print('Epoch: {0:2d}, Batch: {1:8d}, Cost = {2:.6f}, Time: {3:.2f}'.format(
                       epoch, batch_i, loss, time.time() - start))
                start = time.time()

            loss.backward()
            optimizer.step()

        epoch_outdir = outdir + '.' + str(epoch + 1)
        print('Save embeddings into ', epoch_outdir)
        save_embeddings(epoch_outdir)
            
    print('Done')

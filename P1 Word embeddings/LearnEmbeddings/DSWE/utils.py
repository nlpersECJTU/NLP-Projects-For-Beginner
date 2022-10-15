from cProfile import label
import logging
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter



def load_data(conf, dtrain):
    # load data and create vocab
    # words occurred less than 'conf.word_cut_times' are discarded
    
    # words
    wc = Counter()
    # connectives as labels
    cc = set()
    with open(dtrain, 'r') as f:
        for line in tqdm(f):
            arg1, arg2, conn = line.strip().split(' ||| ')
            # ws1 = re.split('[ ]+', arg1)
            # ws2 = re.split('[ ]+', arg2)
            ws1 = arg1.split(' ')
            ws2 = arg2.split(' ')
            wc.update(ws1)
            wc.update(ws2)
            cc.add(conn)
    labels = {c:idx for idx, c in enumerate(cc)}

    # vocab, word -> idx
    wc_dict = { k : v for k, v in wc.items() if v > conf.word_cutoff}
    vocab = { '$PAD$': 0, '$OOV$': 1 }
    i = 2
    for k, v in wc_dict.items():
        vocab[k] = i
        i += 1
    
    # prepare data for training
    x_arg1 = []
    x_arg2 = []
    y      = []
    arg1_len = []
    arg2_len = []

    ml = conf.arg_max_len
    with open(dtrain, 'r') as f:
        for line in tqdm(f):
            arg1, arg2, conn = line.strip().split(' ||| ')
            # arg1 = re.split('[ ]+', arg1)
            # arg2 = re.split('[ ]+', arg2)
            arg1 = arg1.split(' ')
            arg2 = arg2.split(' ')
            
            arg1_len.append(len(arg1))
            arg2_len.append(len(arg2))
            
            t1, t2 = [0] * ml, [0] * ml
            for i, w in enumerate(arg1):
                t1[i] = vocab[w] if w in vocab else vocab['$OOV$']
            for i, w in enumerate(arg2):
                t2[i] = vocab[w] if w in vocab else vocab['$OOV$']

            x_arg1.append(t1)
            x_arg2.append(t2)
            y.append(labels[conn])
    
    # list to nparray
    x_arg1   = np.asarray(x_arg1)
    x_arg2   = np.asarray(x_arg2)
    y        = np.asarray(y)
    arg1_len = np.asarray(arg1_len)
    arg2_len = np.asarray(arg2_len)

    return vocab, (x_arg1, x_arg2, y, arg1_len, arg2_len)



def batch_iter(data, batch_size, shuffle=True):
    x_arg1, x_arg2, y, arg1_len, arg2_len = data
    data_size = x_arg1.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        x_arg1   = x_arg1[shuffled_indices]
        x_arg2   = x_arg2[shuffled_indices]
        y        = y[shuffled_indices]
        arg1_len = arg1_len[shuffled_indices]
        arg2_len = arg2_len[shuffled_indices]

    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_size)

        yield (x_arg1[start_id:end_id], 
               x_arg2[start_id:end_id],
               y[start_id:end_id],  
               arg1_len[start_id:end_id], 
               arg2_len[start_id:end_id])
               


def load_txt_embeddings(fn):
    '''each line: WORD dim_1 dim_2, ..., dim_n'''
    
    vocab = {}
    embeddings = []
    with open(fn, 'r') as f:
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
        


def get_mask(lengths, max_len):
    """
    :param lengths: torch.LongTensor
    :param max_len: int
    :return: mask with shape [len(lengths), max_len]
    """

    m = torch.arange(max_len)[None, :] < lengths[:, None]
    return m.float()



def get_logger(logdir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(logdir, mode='a')
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

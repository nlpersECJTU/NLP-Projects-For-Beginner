import logging
import random
import torch
import numpy as np
from tqdm import tqdm


def load_embedings(conf):
    fl = conf.word_embed
    dim = conf.word_dim
    fn_list = [conf.dtrain, conf.ddev, conf.dtest]
    
    # words in the train/dev/test datasets
    word_set = set()
    for fn in fn_list:
        with open(fn, 'r') as f:
            for line in f:
                _, _, doc, _ = [x.strip() for x in line.strip().split('\t\t')]
                words = doc.split()
                for word in words:
                    word_set.add(word)

    vocab = {'$PAD$': 0}
    embeddings = [np.zeros(dim)]

    # for txt embedding file
    index = 1
    with open(fl, 'r') as f:
        for line in tqdm(f):
            check = line.strip().split()
            if len(check) == 2 or len(check) != dim + 1:
                continue
            line = line.strip().split()
            if line[0] not in word_set:
                continue
            try:
                embedding = [float(s) for s in line[1:]]
            except:
                print(len(line))
            embeddings.append(embedding)
            vocab[line[0]] = index
            index += 1

    return vocab, np.asarray(embeddings)


def load_data(conf, vocab, dataset = 'train'):
    if dataset == 'train':
        fl = conf.dtrain
    if dataset == 'dev':
        fl = conf.ddev
    if dataset == 'test':
        fl = conf.dtest
    max_word_num = conf.max_word_num
    max_sent_num = conf.max_sent_num
    
    data_x = []
    data_y = []

    # read lines from file
    # split each line, each line with format ‘user  product  doc  label’
    with open(fl, 'r') as f:
        for line in tqdm(f):
            line = [x.strip() for x in line.strip().split('\t\t')]
            data_x.append(line[2].lower())
            data_y.append(int(line[3]) - 1)

    # word -> idx
    # cut long sentence/document and pad short sentence/document
    x, sent_len, doc_len = [], [], []
    for _, doc in enumerate(data_x):
        t_sen_len = [0] * max_sent_num
        t_x = np.zeros((max_sent_num, max_word_num), dtype=int)
        sentences = doc.split('<sssss>')
        i = 0
        for sen in sentences:
            j = 0
            for word in sen.strip().split():
                if j >= max_word_num:
                    break
                if word not in vocab:
                    continue
                t_x[i, j] = vocab[word]
                j += 1
            t_sen_len[i] = j
            i += 1
            if i >= max_sent_num:
                break
        doc_len.append(i)
        sent_len.append(t_sen_len)
        x.append(t_x)
    
    # list to nparray
    data_x   = np.asarray(x)
    data_y   = np.asarray(data_y)
    sent_len = np.asarray(sent_len)
    doc_len  = np.asarray(doc_len)
    return data_x, data_y, sent_len, doc_len


def batch_iter(data_set, batch_size, shuffle=True):
    data_x, data_y, sen_lens_x, doc_lens_x = data_set
    data_size = data_x.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        data_x = data_x[shuffled_indices]
        data_y = data_y[shuffled_indices]
        sen_lens_x = sen_lens_x[shuffled_indices]
        doc_lens_x = doc_lens_x[shuffled_indices]

    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_size)
        yield (data_x[start_id:end_id], data_y[start_id:end_id],
               sen_lens_x[start_id:end_id], doc_lens_x[start_id:end_id])


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


def get_mask(lengths, max_len):
    """
    :param lengths: torch.LongTensor
    :param max_len: int
    :return: mask with shape [len(lengths), max_len]
    """

    m = torch.arange(max_len)[None, :] < lengths[:, None]
    return m.float()

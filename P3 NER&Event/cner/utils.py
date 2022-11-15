import os, sys
import logging
import random
import torch
import numpy as np


def get_labels(dataset, strategy = "bio"):
    raw_labels = {'cluener': ['address', 'book', 'company', 'game', 'government', 
                              'movie', 'name', 'organization', 'position', 'scene'],
                  'resume':  ['CONT', 'EDU', 'LOC', 'NAME', 'ORG', 'PRO', 'RACE', 'TITLE'],
                  'weibo':   ['GPE.NAM', 'LOC.NAM', 'LOC.NOM', 'ORG.NAM', 'ORG.NOM', 'PER.NAM', 'PER.NOM']}
    
    if strategy == "bio":
        labels =  ["B-" + lbl for lbl in raw_labels[dataset]]
        labels += ["I-" + lbl for lbl in raw_labels[dataset]]
        labels =  ["O"] + labels
    if strategy == "span":
        labels =  ["O"] + raw_labels[dataset]
    return labels


def load_data_tagging(data_dir, tokenizer, label_list, max_seq_len=128):
    """ Load training, dev or test data, with labels """
    sentences = []
    labels = []
    with open(data_dir, 'r') as fin:
        sen = []
        lbl = []
        for line in fin:
            items = line.strip().split()
            if len(items) == 0:
                sentences.append(sen)
                labels.append(lbl)
                sen = []
                lbl = []
            else:
                sen.append(items[0])
                lbl.append(items[1])

    # tokenize and pad
    data_x = []
    data_y = []
    data_mask = []
    data_segid = []
    data_len = []
    label_map = {label: i for i, label in enumerate(label_list)}
    for ex_ids, (sen, lbl) in enumerate(zip(sentences, labels)):
        sen_seq = " ".join(sen)
        tokens = tokenizer.tokenize(sen_seq)
        lbl_ids = [label_map[x] for x in lbl]
        
        # In the weibo dataset, some specical tokens like '�' are discarded after tokenizing.
        # Here, we discarded the corresponding labels simply.
        if "weibo" in data_dir:
            if len(lbl_ids) > len(tokens):
                lbl_ids = lbl_ids[0: len(tokens)]
        assert len(tokens) == len(lbl_ids), print(sen_seq, tokens, lbl_ids)
        
        # [CLS] + tokens + [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: max_seq_len - special_tokens_count]
            lbl_ids = lbl_ids[: max_seq_len - special_tokens_count]
        
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        lbl_ids = [label_map['O']] + lbl_ids + [label_map['O']]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the max sequence length
        padding_length = max_seq_len - input_len
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        # -1 or 0? 
        # the padding label can be arbitrary when the input_mask is used
        lbl_ids += [-1] * padding_length  

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(lbl_ids) == max_seq_len

        if ex_ids < 0:
            print("Tokens: ", " ".join(tokens))
            print("Input ids: ", " ".join([str(x) for x in input_ids]))
            print("Input mask: ", " ".join([str(x) for x in input_mask]))
            print("Segment ids: ", " ".join([str(x) for x in segment_ids]))
            print("Label ids: ", " ".join([str(x) for x in lbl_ids]))

        data_x.append(input_ids)
        data_y.append(lbl_ids)
        data_mask.append(input_mask)
        data_segid.append(segment_ids)
        data_len.append(input_len)

    # list to nparray
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    data_mask = np.asarray(data_mask)
    data_segid = np.asarray(data_segid)
    data_len = np.asarray(data_len)

    return data_x, data_y, data_mask, data_segid, data_len


def load_data_span(data_dir, tokenizer, label_list, max_seq_len=128):
    """ Load training, dev or test data, with labels """
    sentences = []
    labels = []
    with open(data_dir, 'r') as fin:
        sen = []
        lbl = []
        for line in fin:
            items = line.strip().split()
            if len(items) == 0:
                sentences.append(sen)
                labels.append(lbl)
                sen = []
                lbl = []
            else:
                sen.append(items[0])
                lbl.append(items[1])

    entities = []
    for lbl in labels:
        e = get_entity_bio(lbl)
        entities.append(e)

    # tokenize and pad
    data_x = []
    data_y_start = []
    data_y_end = []
    data_mask = []
    data_segid = []
    data_len = []
    label_map = {label: i for i, label in enumerate(label_list)}
    for ex_ids, (sen, entity) in enumerate(zip(sentences, entities)):
        sen_seq = " ".join(sen)
        tokens = tokenizer.tokenize(sen_seq)
        
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        for e in entity:
            e_lbl = e[0]  # str
            e_start = e[1]
            e_end = e[2]
            start_ids[e_start] = label_map[e_lbl]
            end_ids[e_end] = label_map[e_lbl]

        
        # [CLS] + tokens + [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: max_seq_len - special_tokens_count]
            start_ids = start_ids[: max_seq_len - special_tokens_count]
            end_ids = end_ids[: max_seq_len - special_tokens_count]
        
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        start_ids = [label_map['O']] + start_ids + [label_map['O']]
        end_ids = [label_map['O']] + end_ids + [label_map['O']]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the max sequence length
        padding_length = max_seq_len - input_len
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        start_ids += [0] * padding_length  #  -1 or 0?
        end_ids += [0] * padding_length    #  -1 or 0?

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(start_ids) == max_seq_len
        assert len(end_ids) == max_seq_len

        if ex_ids < 0:
            print("Tokens: ", " ".join(tokens))
            print("Input ids: ", " ".join([str(x) for x in input_ids]))
            print("Input mask: ", " ".join([str(x) for x in input_mask]))
            print("Segment ids: ", " ".join([str(x) for x in segment_ids]))
            print("Entity start ids: ", " ".join([str(x) for x in start_ids]))
            print("Entity end ids: ", " ".join([str(x) for x in end_ids]))

        data_x.append(input_ids)
        data_y_start.append(start_ids)
        data_y_end.append(end_ids)
        data_mask.append(input_mask)
        data_segid.append(segment_ids)
        data_len.append(input_len)


    # list to nparray
    data_x = np.asarray(data_x)
    data_y_start = np.asarray(data_y_start)
    data_y_end = np.asarray(data_y_end)
    data_mask = np.asarray(data_mask)
    data_segid = np.asarray(data_segid)
    data_len = np.asarray(data_len)

    return data_x, data_y_start, data_y_end, data_mask, data_segid, data_len


def batch_iter_tagging(data_set, batch_size, shuffle=True):
    data_x, data_y, data_mask, data_segid, data_len = data_set
    data_size = data_x.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        data_x = data_x[shuffled_indices]
        data_y = data_y[shuffled_indices]
        data_mask = data_mask[shuffled_indices]
        data_segid = data_segid[shuffled_indices]
        data_len = data_len[shuffled_indices]

    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_size)
        # the longest sequence in a batch
        # can save time 
        max_len = max(data_len[start_id:end_id]).item()
        yield (data_x[start_id:end_id, :max_len], 
               data_y[start_id:end_id, :max_len], 
               data_mask[start_id:end_id, :max_len], 
               data_segid[start_id:end_id, :max_len],
               data_len[start_id:end_id])


def batch_iter_span(data_set, batch_size, shuffle=True):
    data_x, data_y_start, data_y_end, data_mask, data_segid, data_len = data_set
    data_size = data_x.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        data_x = data_x[shuffled_indices]
        data_y_start = data_y_start[shuffled_indices]
        data_y_end = data_y_end[shuffled_indices]
        data_mask = data_mask[shuffled_indices]
        data_segid = data_segid[shuffled_indices]
        data_len = data_len[shuffled_indices]
   
    for i in range(num_batches):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_size)
        # the longest sequence in a batch 
        # can save time
        max_len = max(data_len[start_id:end_id]).item()
        yield (data_x[start_id:end_id, :max_len], 
               data_y_start[start_id:end_id, :max_len], 
               data_y_end[start_id:end_id, :max_len], 
               data_mask[start_id:end_id, :max_len], 
               data_segid[start_id:end_id, :max_len],
               data_len[start_id:end_id])


def extract_entity(start, end, seq_len):
    """ Extract entities based on the ture/pred boundary information (start, end), 
        which is only suitable for the flat entities.
    """
    entities = []
    # Two special tokens [CLS], [SEP] are discarded
    start_pred = start.cpu().numpy()[1: seq_len-1]
    end_pred = end.cpu().numpy()[1: seq_len-1]
    
    for i, start_lbl in enumerate(start_pred):
        # lable "O"
        if start_lbl == 0:
            continue
        for j, end_lbl in enumerate(end_pred[i:]):
            # get an entity, the nearest end position
            if start_lbl == end_lbl:
                entities.append((start_lbl, i, i + j))
                break
    return entities



def get_entity_bio(seq, id2label=None):
    """Gets entities from sequence, BIO
        Example:
            # >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC]
            # >>> id2label = ...
            # >>> get_entity_bio(seq)
            [['PER, 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks



# Output information on both Screen and LogFile
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


# Fix seed to get the same results 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

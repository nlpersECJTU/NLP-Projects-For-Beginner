# Some codes are borrowed from 
# https://github.com/elikip/bist-parser/blob/master/bmstparser

import decoder
import torch
import torch.nn as nn


class GraphModel(nn.Module):
    def __init__(self, word_count, words, poss, rels, conf):
        super(GraphModel, self).__init__()
        self.word_count = word_count   # not used
        self.words = words
        self.poss = poss
        self.rels = rels
        self.conf = conf
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(conf.dropout)
        # embeddings
        self.wlookup = nn.Embedding(len(words), conf.wdims)
        self.plookup = nn.Embedding(len(poss), conf.pdims)
        self.rlookup = nn.Embedding(len(rels), conf.rdims)
        
        lstm_input_size = conf.wdims + conf.pdims
        self.lstm = nn.LSTM(lstm_input_size, 
                            conf.lstm_hidden_size // 2, 
                            num_layers = conf.lstm_layers,
                            batch_first = True,
                            bidirectional= True)
        # for tree
        self.hmlp = nn.Linear(conf.lstm_hidden_size, conf.hidden_size)
        self.tmlp = nn.Linear(conf.lstm_hidden_size, conf.hidden_size)
        self.out_layer = nn.Linear(conf.hidden_size, 1)
        # for relation labels
        self.rhmlp = nn.Linear(conf.lstm_hidden_size, conf.hidden_size)
        self.rtmlp = nn.Linear(conf.lstm_hidden_size, conf.hidden_size)
        self.rout_layer = nn.Linear(conf.hidden_size, len(rels))


    def _word_pair_score(self, lstm_out):
        hmlp_out = self.hmlp(lstm_out).squeeze(0)
        tmlp_out = self.tmlp(lstm_out).squeeze(0)

        dim0 = hmlp_out.size(0)
        dim1 = hmlp_out.size(0)
        dim2 = hmlp_out.size(-1)
        h_t = torch.zeros(dim0, dim1, dim2)
        for i in range(dim0):
            for j in range(dim1):
                h_t[i][j] = hmlp_out[i] + tmlp_out[j]
        scores = self.out_layer(self.activation(h_t)).reshape(dim0, -1)
        
        return scores


    def _relation_score(self, lstm_out, heads):
        rhmlp_out = self.rhmlp(lstm_out).squeeze(0)
        rtmlp_out = self.rtmlp(lstm_out).squeeze(0)
        dim0 = len(heads) - 1
        dim1 = rhmlp_out.size(-1)
        
        r_rep = torch.zeros(dim0, dim1)
        for t, h in enumerate(heads[1:]):
            r_rep[t] = rtmlp_out[t+1] + rhmlp_out[h]
        r_scores = self.rout_layer(self.activation(r_rep))

        return r_scores
    
    
    def forward(self, sentence, gold_heads=None):
        # one sentence per batch
        word_ids = []
        pos_ids = []
        for entry in sentence:
            word_ids.append(self.words.get(entry.norm, 0))
            pos_ids.append(self.poss.get(entry.pos, 0))

        word_embs = self.wlookup(torch.LongTensor(word_ids))
        pos_embs = self.plookup(torch.LongTensor(pos_ids))
        inputs = torch.cat([word_embs, pos_embs], dim=-1)
        lstm_out, (hn, cn) = self.lstm(inputs.unsqueeze(0))
        lstm_out = self.dropout(lstm_out)

        # scores for word pairs
        head_scores = self._word_pair_score(lstm_out)
        pred_heads = decoder.parse_proj(head_scores.detach().numpy(), gold_heads)
        
        # scores for relations
        # gold heads for train, the predicted heads for test
        heads = gold_heads or pred_heads
        r_scores = self._relation_score(lstm_out, heads)
        
        return head_scores, r_scores, pred_heads
        

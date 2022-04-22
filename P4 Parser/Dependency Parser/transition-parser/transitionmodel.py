# Some codes are borrowed from 
# https://github.com/elikip/bist-parser/blob/master/barchybrid

from operator import itemgetter
import torch
import torch.nn as nn


class TranModel(nn.Module):
    def __init__(self, word_count, words, poss, rels, conf):
        super(TranModel, self).__init__()
        self.word_count = word_count   # not used
        self.words = words
        self.poss = poss
        self.rels = rels
        self.conf = conf
        self.window = conf.window
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(conf.dropout)
        # embeddings
        self.wlookup = nn.Embedding(len(words), conf.wdims)
        self.plookup = nn.Embedding(len(poss), conf.pdims)
        self.rlookup = nn.Embedding(len(rels), conf.rdims)
        # bilstm
        lstm_input_size = conf.wdims + conf.pdims
        self.lstm = nn.LSTM(lstm_input_size, 
                            conf.lstm_hidden_size // 2, 
                            num_layers = conf.lstm_layers,
                            batch_first = True,
                            bidirectional= True)
        
        # for three actions, self.window words in stack and one word in buffer
        hid_input_size = conf.lstm_hidden_size * (self.window + 1)
        self.hidlayer = nn.Linear(hid_input_size, conf.hidden_size)
        self.outlayer = nn.Linear(conf.hidden_size, 3)
        # for relations, 
        # relations for left arcs, relations for right arcs, and no rel for shift action
        self.rhidlayer = nn.Linear(hid_input_size, conf.hidden_size)
        self.routlayer = nn.Linear(conf.hidden_size, 2 * len(rels) + 1)
        
        # store outputs of the lstm layer
        self.lstm_out = None
        # padding vector when less than self.window words
        self.padding = nn.Parameter(torch.zeros(conf.lstm_hidden_size))
        # self.irels = {v:k for k, v in rels.items()}


    def cal_action_scores(self, stack, buf, istrain=True):
        # prepare input 
        # the last self.window words in stack and one word in buffer 
        top_stack = [self.lstm_out[stack.roots[-i-1].id] if len(stack) > i else self.padding 
                     for i in range(self.window)]
        top_buffer = [self.lstm_out[buf.roots[0].id] if len(buf) > 0 else self.padding]
        input = torch.cat(top_stack + top_buffer)

        # calculate scores for actions and relations, respectively
        uscrs = self.outlayer(self.activation(self.hidlayer(input)))
        scrs = self.routlayer(self.activation(self.rhidlayer(input)))
        # data w/o grad
        uscrs_d = uscrs.detach().numpy()
        scrs_d  = scrs.detach().numpy()

        # transition conditions
        # s0, s1, b0
        # left  arc (0): remove s0, add arc s0<--b0 with relation l
        # right arc (1): remove s0, add arc s1-->s0 with relation l
        # shift     (2): move b0 from buffer to stack
        left_arc_cond = len(stack) > 0 and len(buf) > 0
        right_arc_cond = len(stack) > 1 and stack.roots[-1].id != 0
        shift_cond = len(buf) > 0 and buf.roots[0].id != 0

        # score = action score + relation score,
        # return all scores of actions when training,
        # return the max score for each action when test,
        # (relation, action_id, score data w/o grad, score)
        if istrain:
            scores = [[(rel, 0, uscrs_d[1] + scrs_d[1 + j * 2], uscrs[1] + scrs[1 + j * 2]) for rel, j in self.rels.items()] if left_arc_cond else [],
                      [(rel, 1, uscrs_d[2] + scrs_d[2 + j * 2], uscrs[2] + scrs[2 + j * 2]) for rel, j in self.rels.items()] if right_arc_cond else [],
                      [(None, 2, uscrs_d[0] + scrs_d[0], uscrs[0] + scrs[0])] if shift_cond else []]
        else:
            scores = [[max([(rel, 0, uscrs_d[1] + scrs_d[1 + j * 2]) for rel, j in self.rels.items()], key = itemgetter(2))] if left_arc_cond else [],
                      [max([(rel, 1, uscrs_d[2] + scrs_d[2 + j * 2]) for rel, j in self.rels.items()], key = itemgetter(2))] if right_arc_cond else [],
                      [(None, 2, uscrs_d[0] + scrs_d[0])] if shift_cond else []]

        return scores


    def forward(self, sentence):
        # one sentence per batch
        word_ids = []
        pos_ids = []
        for entry in sentence:
            word_ids.append(self.words.get(entry.norm, 0))
            pos_ids.append(self.poss.get(entry.pos, 0))
        
        # load word embeddings
        word_embs = self.wlookup(torch.LongTensor(word_ids))
        pos_embs = self.plookup(torch.LongTensor(pos_ids))
        inputs = torch.cat([word_embs, pos_embs], dim=-1)
        
        # cal contextual word representations
        self.lstm_out, (hn, cn) = self.lstm(inputs.unsqueeze(0))
        self.lstm_out = self.dropout(self.lstm_out.squeeze(0))

        return self.lstm_out
        
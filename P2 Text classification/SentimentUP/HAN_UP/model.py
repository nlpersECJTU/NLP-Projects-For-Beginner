import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class EmbedLayer(nn.Module):
    def __init__(self, we_tensor, finetune=False):
        super(EmbedLayer, self).__init__()
        self.embed = nn.Embedding(we_tensor.shape[0], we_tensor.shape[1])
        self.embed.weight.data.copy_(we_tensor)
        self.embed.weight.requires_grad = finetune

    def forward(self, x):
        return self.embed(x)



class ProdAttnLayer(nn.Module):
    def __init__(self, inpt_dim, p_dim):
        super(ProdAttnLayer, self).__init__()
        self.attn_inpt = nn.Linear(inpt_dim, inpt_dim, bias=True)
        self.attn_p = nn.Linear(p_dim, inpt_dim, bias=False)
        self.contx = nn.Linear(inpt_dim, 1, bias=False)

    def forward(self, inpt, p, inpt_lens=None):
        inpt_p = torch.tanh_(self.attn_inpt(inpt) + self.attn_p(p))
        a = self.contx(inpt_p)
        # mask
        if inpt_lens is not None:
            m = inpt_lens[:, :, None]
            a = a - (1 - m) * 1e31
        a = F.softmax(a, dim=1)
        s = (a * inpt).sum(1)

        return s, a.transpose(1, 2)



class UserTranLayer(nn.Module):
    def __init__(self, inpt_dim, u_dim, dropout=0.0):
        super(UserTranLayer, self).__init__()
        self.tran_linear1 = nn.Linear(inpt_dim + u_dim, inpt_dim)
        self.tran_linear2 = nn.Linear(inpt_dim, inpt_dim)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, inpt, u):
        h = self.drop(self.relu(self.tran_linear1(torch.cat([inpt, u], dim=-1))) + inpt)
        h = self.drop(self.relu(self.tran_linear2(h)) + h)

        return h



class HAN_UP(nn.Module):
    def __init__(self, conf, we):
        super(HAN_UP, self).__init__()
        self.conf = conf

        # random initialize user and prod as uniform distribution (-0.01, 0.01)
        self.user_embed = nn.Embedding(conf.user_num, conf.user_dim)
        nn.init.uniform_(self.user_embed.weight, -0.01, 0.01)

        self.prod_embed = nn.Embedding(conf.prod_num, conf.prod_dim)
        nn.init.uniform_(self.prod_embed.weight, -0.01, 0.01)
        
        # word embeddings
        self.embed_layer = EmbedLayer(we, conf.finetune)

        # word level
        self.word_trans = UserTranLayer(conf.word_dim, conf.user_dim, conf.dropout)
        self.word_norm = LayerNorm(conf.word_dim)
        self.word_encoder = nn.LSTM(conf.word_dim, 
                                    conf.word_hidden_size, 
                                    bidirectional=True, 
                                    batch_first=True)
        self.word_attn = ProdAttnLayer(conf.word_hidden_size * 2, conf.prod_dim)

        # sentence level
        self.sent_trans = UserTranLayer(conf.word_hidden_size * 2, conf.user_dim, conf.dropout)
        self.sent_norm = LayerNorm(conf.word_hidden_size * 2)
        self.sent_encoder = nn.LSTM(conf.word_hidden_size*2,
                                    conf.sent_hidden_size,
                                    bidirectional=True,
                                    batch_first=True)
        self.sent_attn = ProdAttnLayer(conf.sent_hidden_size * 2, conf.prod_dim)

        # document level
        self.doc_trans = UserTranLayer(conf.sent_hidden_size * 2, conf.user_dim, conf.dropout)
        self.doc_norm = LayerNorm(conf.sent_hidden_size * 2)

        # mlp for claasification
        self.mlp = nn.Linear(conf.sent_hidden_size * 2, conf.num_classes)
        self.drop = nn.Dropout(conf.dropout)


    def forward(self, u, p, x, sen_lens, doc_lens):
        conf = self.conf

        # load user and prodct embeddings
        u_embed = self.user_embed(u)
        p_embed = self.prod_embed(p)
        
        # load word embeddings
        x_embed = self.embed_layer(x)
        
        # word -> sentence
        # for shape adaptation
        zeros_u = torch.zeros(x_embed.size(0), x_embed.size(1), x_embed.size(2), conf.user_dim)
        zeros_u = zeros_u.to(self.conf.device)
        zeros_p = torch.zeros(x_embed.size(0), x_embed.size(1), x_embed.size(2), conf.prod_dim)
        zeros_p = zeros_p.to(self.conf.device)

        u_embed_ = zeros_u + u_embed[:, None, None, :]
        u_embed_ = u_embed_.view(-1, u_embed_.size(2), u_embed_.size(3))
        x_embed = x_embed.view(-1, x_embed.size(2), x_embed.size(3))
        
        # transition word embeddings conditioning on user, and normalize them
        x_embed = self.word_trans(x_embed, u_embed_)
        x_embed = self.word_norm(x_embed)
        
        # learn word-level context based on BiLSTM
        word_rep, _ = self.word_encoder(x_embed)

        # shape adaptation
        p_embed_ = zeros_p + p_embed[:, None, None, :]
        p_embed_ = p_embed_.view(-1, p_embed_.size(2), p_embed_.size(3))
        
        # product-biased attention mechanism
        sen_rep,  _ = self.word_attn(word_rep, p_embed_, sen_lens)
        sen_rep     = sen_rep.view([-1, self.conf.max_sent_num, sen_rep.size(1)])
        sen_rep     = self.drop(sen_rep)

        # sentence -> document
        # for shape adaptation
        zeros_u = torch.zeros(sen_rep.size(0), sen_rep.size(1), conf.user_dim)
        zeros_u = zeros_u.to(self.conf.device)
        zeros_p = torch.zeros(sen_rep.size(0), sen_rep.size(1), conf.prod_dim)
        zeros_p = zeros_p.to(self.conf.device)

        # transition sentence representaions conditioning on user
        # and normalize them
        u_embed_ = zeros_u + u_embed[:, None, :]
        sen_rep = self.sent_trans(sen_rep, u_embed_)
        sen_rep = self.sent_norm(sen_rep)

        # learn sentence-level context based on BiLSTM
        sen_rep, _ = self.sent_encoder(sen_rep)

        # product-biased attention mechanism
        p_embed_ = zeros_p + p_embed[:, None, :]
        doc_rep, _ = self.sent_attn(sen_rep, p_embed_, doc_lens)

        # transition sentence representaions conditioning on user
        doc_rep = self.doc_trans(doc_rep, u_embed)
        # and normalize them, needed?
        doc_rep = self.doc_norm(doc_rep)
        doc_rep = self.drop(doc_rep)

        # classification
        logits = self.mlp(doc_rep)

        return logits

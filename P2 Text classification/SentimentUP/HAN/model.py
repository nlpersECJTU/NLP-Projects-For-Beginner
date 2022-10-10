import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedLayer(nn.Module):
    def __init__(self, we_tensor, finetune=False):
        super(EmbedLayer, self).__init__()
        self.embed = nn.Embedding(we_tensor.shape[0], we_tensor.shape[1])
        self.embed.weight.data.copy_(we_tensor)
        self.embed.weight.requires_grad = finetune

    def forward(self, x):
        return self.embed(x)


class AttnLayer(nn.Module):
    def __init__(self, inpt_dim):
        super(AttnLayer, self).__init__()

        self.attn = nn.Linear(inpt_dim, inpt_dim, bias=True)
        self.contx = nn.Linear(inpt_dim, 1, bias=False)

    def forward(self, inpt, inpt_lens=None):

        inpt_ = torch.tanh_(self.attn(inpt))
        a = self.contx(inpt_)
        # mask
        if inpt_lens is not None:
            m = inpt_lens[:, :, None]
            a = a - (1-m) * 1e31
        a = F.softmax(a, dim=1)
        s = (a * inpt).sum(1)
        return s, a.transpose(1, 2)



class HAN(nn.Module):
    def __init__(self, conf, we):
        super(HAN, self).__init__()
        self.conf = conf
        
        self.embed_layer = EmbedLayer(we, conf.finetune)
        self.word_encoder = nn.LSTM(conf.word_dim, 
                                    conf.word_hidden_size, 
                                    bidirectional=True, 
                                    batch_first=True)
        self.word_attn = AttnLayer(conf.word_hidden_size * 2)

        self.sent_encoder = nn.LSTM(conf.word_hidden_size*2,
                                    conf.sent_hidden_size,
                                    bidirectional=True,
                                    batch_first=True)
        self.sent_attn = AttnLayer(conf.sent_hidden_size * 2)

        self.mlp = nn.Linear(conf.sent_hidden_size * 2, conf.num_classes)
        self.drop = nn.Dropout(conf.dropout)

    def forward(self, x, sen_lens, doc_lens):
        # load word embeddings
        embed = self.embed_layer(x)
        embed = embed.view(-1, embed.size(2), embed.size(3))
        
        # word -> sentence
        word_rep, _ = self.word_encoder(embed)
        sen_rep,  _ = self.word_attn(word_rep, sen_lens)
        sen_rep     = sen_rep.view([-1, self.conf.max_sent_num, sen_rep.size(1)])
        sen_rep     = self.drop(sen_rep)

        # sentence -> document
        sen_rep, _ = self.sent_encoder(sen_rep)
        doc_rep, _ = self.sent_attn(sen_rep, doc_lens)
        doc_rep    = self.drop(doc_rep)

        # classification
        logits = self.mlp(doc_rep)

        return logits

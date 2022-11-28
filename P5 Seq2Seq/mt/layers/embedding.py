# by wucx
# 2022-11-17

import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, pad_idx=1):
        super(TokenEmbedding, self).__init__()
        # should padding_idx be a parameter?
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding
    """

    def __init__(self, d_model, max_len, device) -> None:
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.pe = torch.zeros(max_len, d_model, device=device)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        # compute the positional encodings once in log space
        # less time-consuming
        div_term = torch.exp(torch.arange(0, d_model, step=2, device=device) * 
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)


    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len, :]



class TransformerEmbedding(nn.Module):
    """
    token embedding + positional embedding
    positional embedding can give positional informaiton to network
    """

    def __init__(self, vocab_size, d_model, pad_idx, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)

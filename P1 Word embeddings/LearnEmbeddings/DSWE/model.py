import torch
import torch.nn as nn

class AvgModel(nn.Module):
    def __init__(self, conf):
        super().__init__()

        # embedding layer
        self.embed   = nn.Embedding(conf.word_num, conf.word_dim)
        nn.init.uniform_(self.embed.weight, -0.01, 0.01)
        
        # two linear layers
        self.linear1 = nn.Linear(conf.word_dim * 2, conf.hid_size)
        self.linear2 = nn.Linear(conf.hid_size, conf.class_num)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(conf.dropout)


    def forward(self, arg1, arg2, arg1_lens=None, arg2_lens=None):
        arg1_embed = self.embed(arg1)
        arg2_embed = self.embed(arg2)

        if arg1_lens is None:
            arg1_avg = torch.mean(arg1_embed, dim=1)
        else:
            lens1 = arg1_lens[:, :, None]
            arg1_mask = arg1_embed * lens1
            arg1_sum = torch.sum(arg1_mask, dim=1)
            arg1_avg = arg1_sum / torch.sum(arg1_lens, dim=-1)[:, None]
        
        if arg2_lens is None:
            arg2_avg = torch.mean(arg2_embed, dim=1)
        else:
            lens2 = arg2_lens[:, :, None]
            arg2_mask = arg2_embed * lens2
            arg2_sum = torch.sum(arg2_mask, dim=1)
            arg2_avg = arg2_sum / torch.sum(arg2_lens, dim=-1)[:, None]

        arg_avg = torch.concat([arg1_avg, arg2_avg], dim=-1)
        arg_avg = self.drop(arg_avg)

        # two layers
        arg_rep = self.linear1(arg_avg)
        arg_rep = self.relu(arg_rep)
        o = self.linear2(arg_rep)

        return o

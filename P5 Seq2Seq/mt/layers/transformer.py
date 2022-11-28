# by wucx
# 2022-11-17

from torch import nn
from layers.embedding import TransformerEmbedding
from layers.sublayer import EncoderLayer, DecoderLayer
from utils import make_pad_mask, make_tril_mask

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, src_pad_idx, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        pad_idx=src_pad_idx,
                                        drop_prob=drop_prob,
                                        device=device)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                    for _ in range(n_layers)])        

    def forward(self, x, s_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, tgt_pad_idx, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        pad_idx=tgt_pad_idx,
                                        drop_prob=drop_prob,
                                        device=device)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)


    def forward(self, tgt, enc_src, tgt_mask, tgt_src_mask):
        tgt = self.emb(tgt)
        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, tgt_src_mask)

        # pass to LM head
        output = self.linear(tgt)
        return output


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx, enc_voc_size, dec_voc_size, 
                       d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.dec_voc_size = dec_voc_size
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               src_pad_idx=src_pad_idx,
                               n_layers=n_layers,
                               drop_prob=drop_prob,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               tgt_pad_idx=tgt_pad_idx,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)


    def forward(self, src, tgt):
        src_mask = make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        tgt_src_mask = make_pad_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx) * make_tril_mask(tgt, tgt).to(self.device)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, tgt_src_mask)
        return output

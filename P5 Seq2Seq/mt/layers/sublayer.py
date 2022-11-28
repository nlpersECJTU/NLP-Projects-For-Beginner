# by wucx
# 2022-11-16


from torch import nn
from layers.ffn import PositionwiseFFN
from layers.attention import MultiHeadAttn
from layers.layernorm import LayerNorm


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.mattn = MultiHeadAttn(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFFN(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        x_ = x
        x = self.mattn(query=x, key=x, value=x, mask=s_mask)

        # 2. add and norm
        x = self.norm1(x + x_)
        x = self.dropout1(x)

        # 3. positionwise feed forward network
        x_ = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x + x_)
        x = self.dropout2(x)
        
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.dec_attn = MultiHeadAttn(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.dec_enc_attn = MultiHeadAttn(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFFN(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, t_s_mask):
        # 1. compute self attention of tgt
        x_ = dec
        x = self.dec_attn(query=dec, key=dec, value=dec, mask=t_mask)

        # 2. add and norm
        x = self.norm1(x + x_)
        x = self.dropout1(x)

        if enc is not None:
            # 3. compute dec(tgt)-enc(src) attention
            x_ = x
            x = self.dec_enc_attn(query=x, key=enc, value=enc, mask=t_s_mask)

            # 4. add and norm
            x = self.norm2(x + x_)
            x = self.dropout2(x)

        # 5. positionwise feed forward network
        x_ = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.norm3(x + x_)
        x = self.dropout3(x)
        return x

# by wucx
# 2022-11-16

import math
import torch
from torch import nn


class ScaleDotProductAttn(nn.Module):
    """
    computer scale dot product attention
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # inputs are 4-dimension tensor
        # [batch_size, head, length, d_tensor]
        d_key = key.size(-1)
        
        # 1. compute similarity
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_key)
        
        # 2. apply masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e16)
        
        # 3. to make [0, 1] range 
        p_attn = self.softmax(scores)

        # 4. multiply with value
        return torch.matmul(p_attn, value), p_attn 


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_tensor = d_model // n_head

        self.attention = ScaleDotProductAttn()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # 1. dot product with weight matrices
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2. split tensor by number of heads
        batch_size = query.size(0)
        q = q.view(batch_size, -1, self.n_head, self.d_tensor).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.d_tensor).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.d_tensor).transpose(1, 2)

        # 3. get attention values
        # sometiems, we need visualize the attention scores
        out, self.attn = self.attention(q, k, v, mask=mask)

        # 4. concatenate and pass to linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_tensor)
        out = self.w_concat(out)

        return out

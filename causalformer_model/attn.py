import math
import torch
import torch.nn as nn
import numpy as np

class FullAttention(nn.Module):
    def __init__(
        self,
        scale=None,
        attention_dropout=0.
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, output_attn=False):
        B, L, H, E = queries.shape  # batch size, sequence len, number of heads, dim_query
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # batch size, number of heads, sequence len, sequence len
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)  # batch size, sequence len, number of heads, dim_value

        if output_attn:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_queries_keys,
        d_values,
        n_heads,
        dropout_qkv=0.0,
    ):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention()
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, output_attn=False):
        B, L, _ = queries.shape  # batch size, seq len, dim_model
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1)
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            output_attn=output_attn,
        )

        out = out.view(B, L, -1)

        if not output_attn:
            assert attn is None

        out = self.out_projection(out)
        return out, attn  # out: (batch size, sequence len, d_model), attn: (batch size, number of heads, sequence len, sequence len)

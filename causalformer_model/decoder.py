import torch
import torch.nn as nn
import torch.nn.functional as F

from extra_layers import (
    Normalization,
    Localize,
    ReverseLocalize,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        global_self_attention,
        local_self_attention,
        global_cross_attention,
        local_cross_attention,
        d_model,
        d_yt,
        d_yc,
        d_ff=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
        norm="layer",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.local_self_attention = local_self_attention
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention
        self.local_cross_attention = local_cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = Normalization(method=norm, d_model=d_model)
        self.norm5 = Normalization(method=norm, d_model=d_model)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_yt = d_yt
        self.d_yc = d_yc  # d_yc = d_yt

    def forward(
        self, x, cross, output_self_attn=False, output_cross_attn=False
    ):  # x: (batch size, seq len x dim_yt, d_model)
        # pre-norm Transformer architecture
        local_self_attns, global_self_attns, local_cross_attns, global_cross_attns = None, None, None, None
        if self.local_self_attention:
            # print('decoder local self attn')
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_yt)
            x1, local_self_attns = self.local_self_attention(x1, x1, x1, output_attn=output_self_attn)
            x1 = ReverseLocalize(x1, self.d_yt)
            x = x + self.dropout_attn_out(x1)

        if self.global_self_attention:
            # print('decoder global self attn')
            x1 = self.norm2(x)
            x1, global_self_attns = self.global_self_attention(
                x1,
                x1,
                x1,
                output_attn=output_self_attn
            )

            x = x + self.dropout_attn_out(x1)

        if self.local_cross_attention:
            # print('decoder local cross attn')
            x1 = self.norm3(x)
            bs, *_ = x1.shape
            x1 = Localize(x1, self.d_yt)
            cross_local = Localize(cross, self.d_yc)[: self.d_yt * bs]
            x1, local_cross_attns = self.local_cross_attention(
                x1,
                cross_local,
                cross_local,
                output_attn = output_cross_attn
            )
            x1 = ReverseLocalize(x1, self.d_yt)
            x = x + self.dropout_attn_out(x1)

        if self.global_cross_attention:
            # print('decoder global cross attn')
            x1 = self.norm4(x)
            x1, global_cross_attns = self.global_cross_attention(
                x1, # batch size, seq_len x dim_y_target, d_model
                cross, # batch size, seq_len x dim_y, d_model
                cross,
                output_attn=output_cross_attn
            )

            x = x + self.dropout_attn_out(x1)  # batch size, d_y, d_model

        x1 = self.norm5(x)  # batch size, d_y, d_model
        # feedforward layers as 1x1 convs
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))  # batch size, d_ff, d_y
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))  # batch size, d_y, d_model
        output = x + x1
        return output, (local_self_attns, global_self_attns, local_cross_attns, global_cross_attns)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, emb_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self,
        val_time_emb,
        space_emb,
        cross,
        output_self_attn=False,
        output_cross_attn=False,
    ):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = [] # a list of size layers. Each element is a tuple (local_self_attns, global_self_attns, local_cross_attns, global_cross_attns)
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                cross,
                output_self_attn=output_self_attn,
                output_cross_attn=output_cross_attn,
            )
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns   # x: (batch size, d_y, d_model)

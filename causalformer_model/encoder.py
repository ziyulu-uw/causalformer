import torch
import torch.nn as nn
import torch.nn.functional as F

from extra_layers import (
    Flatten,
    Normalization,
    Localize,
    ReverseLocalize,
)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        global_attention,
        local_attention,
        d_model,
        d_yc,  # dim y context
        d_ff=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
        norm="layer",
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.local_attention = local_attention
        self.global_attention = global_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_yc = d_yc

    def forward(self, x, output_attn=False):
        # x: (batch_size, dim_y * seq_len, dim)
        # uses pre-norm Transformer architecture
        local_attns, global_attns = None, None
        if self.local_attention:
            # print('encoder local self attn')
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_yc)  # (dim_y * batch_size, seq_len, dim)
            x1, local_attns = self.local_attention(
                x1,
                x1,
                x1,
                output_attn=output_attn
            )
            x1 = ReverseLocalize(x1, self.d_yc)  # (batch_size, dim_y * seq_len, dim)
            x = x + self.dropout_attn_out(x1)

        if self.global_attention:
            # print('encoder global self attn')
            x1 = self.norm2(x)

            x1, global_attns = self.global_attention(
                x1,
                x1,
                x1,
                output_attn=output_attn
            )

            x = x + self.dropout_attn_out(x1)

        x1 = self.norm3(x)
        # feedforward layers (done here as 1x1 convs)
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))
        output = x + x1  # batch size, seq_len x dim_y, d_model

        return output, (local_attns, global_attns)


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers,
        norm_layer,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, val_time_emb, space_emb, output_attn=False):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = []  # a list of size attn_layers. Each element is a tuple (local_attns, global_attns)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, output_attn=output_attn)
            if len(self.conv_layers) > i:
                if self.conv_layers[i] is not None:
                    x = self.conv_layers[i](x)
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns


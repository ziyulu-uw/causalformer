from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd
from einops import rearrange, repeat

from extra_layers import Normalization, FoldForPred
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attn import (
    FullAttention,
    AttentionLayer,
)
from embed import SpatioTemporalEmbedding


class Causalformer(nn.Module):
    def __init__(
        self,
        d_yc: int = 1,  # dim y context
        d_yt: int = 1,  # dim y target (d_yt=d_yc)
        d_x: int = 4,  # dim x  (time)
        max_seq_len: int = None,
        d_model: int = 200,  # Transformer embedding dimension
        d_queries_keys: int = 30,  # dim of queries & keys
        d_values: int = 30,   # dim of values
        n_heads: int = 8,  # number of attention heads
        e_layers: int = 2,  # number of encoder layers
        d_layers: int = 3,  # number of decoder layers
        d_ff: int = None,  # dim of Transformer up-scaling MLP layer. If None, then set to 4 * d_model by default
        time_emb_dim: int = 6,  # dim of time embedding
        dropout_emb: float = 0.1,  # embedding dropout rate. Drop out elements of the embedding vectors during training
        dropout_attn_matrix: float = 0.0,  # attention dropout rate. Dropout elements of the attention matrix. Only applicable to attn mechanisms that explicitly compute the attn matrix (e.g. Full).
        dropout_attn_out: float = 0.0,
        dropout_ff: float = 0.2,  # standard dropout applied to activations of FF networks in the Transformer
        dropout_qkv: float = 0.0,  # query, key and value dropout rate. Dropout elements of these attention vectors during training
        pos_emb_type: str = "t2v",  # position embedding type ("t2v" or "abs")
        enc_global_self_attn: str = "none",  # attention mechanism type ("full" or "prob" or "performer" or "none")
        enc_local_self_attn: str = "full",  # attention mechanism type
        dec_global_self_attn: str = "none",  # attention mechanism type
        dec_local_self_attn: str = "none",  # attention mechanism type
        dec_global_cross_attn: str = "full",  # attention mechanism type
        dec_local_cross_attn: str = "full",  # attention mechanism type
        activation: str = "gelu",  # activation function for Transformer encoder and decoder layers ("relu" or "gelu")
        norm: str = "batch",  # normalization method ("layer" or "batch" or "scale" or "power" or "none")
        use_final_norm: bool = False,
        device=torch.device("cuda:0"),  # torch.device("mps"), torch.device("cuda:0"), torch.device("cpu")
        out_dim: int = 1,
        finalact = None,
    ):
        super().__init__()

        self.d_yt = d_yt
        self.d_yc = d_yc
        self.device = device
        self.finalact = finalact


        # embeddings.
        self.enc_embedding = SpatioTemporalEmbedding(
            d_y=d_yc,
            d_x=d_x,
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            position_emb=pos_emb_type,
            max_seq_len=max_seq_len,
        )


        # Select Attention Mechanisms
        attn_kwargs = {
            "d_model": d_model,
            "n_heads": n_heads,
            "d_qk": d_queries_keys,
            "d_v": d_values,
            "dropout_qkv": dropout_qkv,
            "dropout_attn_matrix": dropout_attn_matrix,
        }

        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    global_attention=self._attn_switch(
                        enc_global_self_attn,
                        **attn_kwargs,
                    ),
                    local_attention=self._attn_switch(
                        enc_local_self_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    d_yc=d_yc,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(e_layers)
            ],

            conv_layers=[],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    global_self_attention=self._attn_switch(
                        dec_global_self_attn,
                        **attn_kwargs,
                    ),
                    local_self_attention=self._attn_switch(
                        dec_local_self_attn,
                        **attn_kwargs,
                    ),
                    global_cross_attention=self._attn_switch(
                        dec_global_cross_attn,
                        **attn_kwargs,
                    ),
                    local_cross_attention=self._attn_switch(
                        dec_local_cross_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    d_yt=d_yt,
                    d_yc=d_yc,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(d_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )


        # final linear layers turn Transformer output into predictions
        self.forecaster_onestep = nn.Linear(d_model, out_dim, bias=True)

    def forward(
        self,
        x_c,
        y_c,
        x_t,
        y_t,
        output_enc_attention=False,
        output_dec_self_attention=False,
        output_dec_cross_attention=False
    ):
        # set data to [batch, length, dim] format
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        enc_x = x_c.to(self.device).float()
        enc_y = y_c.to(self.device).float()
        dec_x = x_t.to(self.device).float()
        # embed context sequence
        enc_vt_emb, enc_s_emb = self.enc_embedding(
            y=enc_y, x=enc_x,
        )

        # encode context sequence
        enc_out, enc_attns = self.encoder(
            val_time_emb=enc_vt_emb,
            space_emb=enc_s_emb,
            output_attn=output_enc_attention,
        )

        # zero out target sequence
        dec_y = torch.zeros_like(y_t).to(self.device)
        # embed target sequence
        dec_vt_emb, dec_s_emb = self.enc_embedding(y=dec_y, x=dec_x)

        # decode target sequence w/ encoded context
        dec_out, dec_attns = self.decoder(
            val_time_emb=dec_vt_emb,
            space_emb=dec_s_emb,
            cross=enc_out,
            output_self_attn=output_dec_self_attention,
            output_cross_attn=output_dec_cross_attention
        )

        # forecasting predictions
        if self.finalact is None:
            onestepforecast_out = self.forecaster_onestep(dec_out)
        else:
            onestepforecast_out = self.finalact(self.forecaster_onestep(dec_out))

        # fold flattened spatiotemporal format back into (batch, length, d_yt)
        onestepforecast_out = FoldForPred(onestepforecast_out, dy=self.d_yt)

        return (onestepforecast_out,
                (enc_attns, dec_attns))

    def _attn_switch(
        self,
        attn_str: str,
        d_model: int,
        n_heads: int,
        d_qk: int,
        d_v: int,
        dropout_qkv: float,
        dropout_attn_matrix: float,
    ):

        if attn_str == "full":
            # standard full (n^2) attention
            Attn = AttentionLayer(
                attention=partial(FullAttention, attention_dropout=dropout_attn_matrix),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                dropout_qkv=dropout_qkv,
            )

        elif attn_str.lower() == "none":
            Attn = None
        return Attn

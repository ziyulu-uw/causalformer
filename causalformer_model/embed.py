import torch
import torch.nn as nn
from time2vec import Time2Vec
from einops import rearrange, repeat
from extra_layers import Flatten

class SpatioTemporalEmbedding(nn.Module):
    def __init__(
        self,
        d_y,
        d_x,
        d_model,
        time_emb_dim=6,
        position_emb="abs",
        max_seq_len=None,
    ):
        super().__init__()

        time_dim = time_emb_dim * d_x
        self.time_emb = Time2Vec(d_x, embed_dim=time_dim)

        assert position_emb in ["t2v", "abs"]
        self.max_seq_len = max_seq_len
        self.position_emb = position_emb
        if self.position_emb == "t2v":
            # standard periodic pos emb but w/ learnable coeffs
            self.local_emb = Time2Vec(1, embed_dim=d_model + 1)
        elif self.position_emb == "abs":
            # lookup-based learnable pos emb
            assert max_seq_len is not None
            self.local_emb = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=d_model
            )

        y_emb_inp_dim = 1
        self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim, d_model)
        self.space_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)
        self.d_model = d_model


    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return self.spatio_temporal_embed(y=y, x=x)

    def spatio_temporal_embed(self, y: torch.Tensor, x: torch.Tensor):  # given is of size dy
        # full spatiotemopral emb method. lots of shape rearrange code
        # here to create artifically long (length x dim) spatiotemporal sequence
        batch, length, dy = y.shape  #batch size, seq len, dim_y

        # position emb ("local_emb") (i.e., relative position in the sequence)
        local_pos = repeat(
            torch.arange(length).to(x.device), f"length -> {batch} ({dy} length)"
        )
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[:, :, 1:]  # (batch size, seq_len x dim_y, d_model)
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())

        # time emb
        # x = torch.nan_to_num(x)
        x = repeat(x, f"batch len x_dim -> batch ({dy} len) x_dim")
        time_emb = self.time_emb(x)  # (batch size, seq_len x dim_y, time_emb_dim * d_x) = (batch size, seq_len x dim_y, time_dim)

        y = Flatten(y)

        # concat time_emb, y --> FF --> val_time_emb
        val_time_inp = torch.cat((time_emb, y), dim=-1)  #(batch size, seq len x dim_y, time_dim + 1)
        val_time_emb = self.val_time_emb(val_time_inp)  #(batch size, seq len x dim_y, d_model)
        val_time_emb = local_emb + val_time_emb  #(batch size, seq len x dim_y, d_model)

        # space embedding
        var_idx = repeat(torch.arange(dy).long().to(x.device), f"dy -> {batch} (dy {length})") #(batch size, seq len x dim_y)
        space_emb = self.space_emb(var_idx)  #(batch size, seq len x dim_y, d_model)

        return val_time_emb, space_emb


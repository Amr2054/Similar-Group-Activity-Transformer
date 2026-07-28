import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class PlayEncoder(nn.Module):
    """
    Two-stage baseline:
      1) Frame encoder: a standard TransformerEncoder over the 23 player
         tokens of a single frame, pooled via a learnable [CLS] token ->
         one embedding per frame.
      2) Play encoder: a standard TransformerEncoder over the sequence of
         frame embeddings (with learned positional embeddings), pooled via
         a learnable [CLS] token -> one embedding per play.
    """

    def __init__(self, input_dim=9, d_model=128, n_heads=4,
                 frame_layers=2, play_layers=2, output_dim=64,
                 max_frames=100, dropout=0.1,
                 pos_encoding="learned"): 
        super().__init__()
        self.d_model = d_model

        self.pos_encoding_type = pos_encoding
        
        # Raw player features -> d_model
        self.player_proj = nn.Linear(input_dim, d_model)

        # ---- Frame encoder (over players within a frame) ----
        self.frame_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        frame_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=frame_layers)

        # ---- Play encoder (over frames within a play) ----
        self.play_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        if pos_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
            self.sinusoidal_pos = None
        elif pos_encoding == "sinusoidal":
            self.pos_embedding = None
            self.sinusoidal_pos = PositionalEncoding(d_model, max_len=max_frames)
        elif pos_encoding == "none":
            self.pos_embedding = None
            self.sinusoidal_pos = None
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")
        
        play_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.play_encoder = nn.TransformerEncoder(play_layer, num_layers=play_layers)

        # Projection head for the contrastive loss
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, features):
        # features: (B, Frames, Players=23, input_dim=9)
        B, Fr, P, _ = features.shape

        # Channel 4 = visibility. Mask out missing/invisible agents.
        visibility = features[..., 4]
        padding_mask = (visibility == 0)  # (B, Fr, P), True = ignore

        x = self.player_proj(features)               # (B, Fr, P, D)
        x = x.reshape(B * Fr, P, self.d_model)
        pad = padding_mask.reshape(B * Fr, P)

        # Never mask out every player in a frame (would starve the CLS token).
        fully_masked = pad.all(dim=1)
        if fully_masked.any():
            pad = pad.clone()
            pad[fully_masked] = False

        cls = self.frame_cls.expand(B * Fr, -1, -1)
        x = torch.cat([cls, x], dim=1)                # (B*Fr, 1+P, D)
        cls_pad = torch.zeros(B * Fr, 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([cls_pad, pad], dim=1)

        x = self.frame_encoder(x, src_key_padding_mask=pad)
        frame_embeddings = x[:, 0]                     # (B*Fr, D) take CLS output
        frame_embeddings = frame_embeddings.reshape(B, Fr, self.d_model)

        if self.pos_encoding_type == "learned":
            frame_embeddings = frame_embeddings + self.pos_embedding[:, :Fr, :]
        elif self.pos_encoding_type == "sinusoidal":
            frame_embeddings = self.sinusoidal_pos(frame_embeddings)
        
        play_cls = self.play_cls.expand(B, -1, -1)
        seq = torch.cat([play_cls, frame_embeddings], dim=1)  # (B, 1+Fr, D)
        seq = self.play_encoder(seq)
        final_embedding = seq[:, 0]                     # (B, D) play-level embedding

        projected_embedding = self.projection_head(final_embedding)
        return final_embedding, projected_embedding

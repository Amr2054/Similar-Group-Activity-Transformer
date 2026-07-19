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


class HierarchicalPlayEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, frame_layers=2, play_layers=2):
        super().__init__()
        self.d_model = d_model

        # Continuous Spatial Tokenizer
        self.player_proj = nn.Sequential(
            nn.Linear(9, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Frame Encoder (Social/Spatial) - No Positional Encoding
        self.frame_cls = nn.Parameter(torch.randn(1, 1, d_model))
        frame_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=frame_layers)

        # Play Encoder (Temporal)
        self.pos_encoder = PositionalEncoding(d_model)
        self.play_cls = nn.Parameter(torch.randn(1, 1, d_model))
        play_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )
        self.play_encoder = nn.TransformerEncoder(play_layer, num_layers=play_layers)

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64)
        )

    def forward(self, features):
        # features: (Batch, Frames=100, Players=23, 9)
        B, F, P, _ = features.shape

        # Channel 4 = visibility.
        # Mask out agents that are missing/invisible/dropped-out so the
        # frame encoder doesn't attend to fabricated zero-vector tokens.
        visibility = features[..., 4]  # (B, F, P)
        agent_padding_mask = (visibility == 0)  # True = ignore

        player_tokens = self.player_proj(features)
        flat_frames = player_tokens.view(B * F, P, self.d_model) # (B*F,23,512)
        flat_padding_mask = agent_padding_mask.view(B * F, P)  # (B*F, 23)

        cls_mask = torch.zeros(B * F, 1, dtype=torch.bool, device=features.device)
        frame_padding_mask = torch.cat([cls_mask, flat_padding_mask], dim=1)  # (B*F, P+1)

        # Safety: never let a frame mask out every single agent (would starve
        # the CLS token's attention entirely) — unmask as a fallback.
        # TODO: Investigate
        fully_masked = flat_padding_mask.all(dim=1)
        if fully_masked.any():
            frame_padding_mask[fully_masked, 1:] = False

        frame_cls_tokens = self.frame_cls.expand(B * F, -1, -1)
        frame_input = torch.cat([frame_cls_tokens, flat_frames], dim=1)

        frame_out = self.frame_encoder(frame_input, src_key_padding_mask=frame_padding_mask)
        frame_embeddings = frame_out[:, 0, :].view(B, F, self.d_model)

        temporal_seq = self.pos_encoder(frame_embeddings)
        play_cls_tokens = self.play_cls.expand(B, -1, -1)
        temporal_input = torch.cat([play_cls_tokens, temporal_seq], dim=1)
        play_out = self.play_encoder(temporal_input)
        final_embedding = play_out[:, 0, :]
        projected_embedding = self.projection_head(final_embedding)

        return final_embedding, projected_embedding
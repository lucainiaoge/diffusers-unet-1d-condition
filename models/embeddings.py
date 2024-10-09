# Adapted from Huggingface by Tongyu Lu
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/embeddings.py

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate
from diffusers.models.activations import FP32SiLU, get_activation
from diffusers.models.attention_processor import Attention

class TextSeqTimeEmbedding(nn.Module):
    def __init__(self, text_embed_dim: int = 768, seq_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)
        self.text_norm = nn.LayerNorm(time_embed_dim)
        self.seq_proj = nn.Linear(seq_embed_dim, time_embed_dim)

    def forward(self, text_embeds: torch.Tensor, seq_embeds: torch.Tensor):
        # text
        time_text_embeds = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # seq
        time_seq_embeds = self.seq_proj(seq_embeds)

        return time_seq_embeds + time_text_embeds


class SeqTimeEmbedding(nn.Module):
    def __init__(self, seq_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.seq_proj = nn.Linear(seq_embed_dim, time_embed_dim)
        self.seq_norm = nn.LayerNorm(time_embed_dim)

    def forward(self, seq_embeds: torch.Tensor):
        # seq
        time_seq_embeds = self.seq_proj(seq_embeds)
        time_seq_embeds = self.seq_norm(time_seq_embeds)
        return time_seq_embeds

class TextSeqProjection(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        seq_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_seq_text_embeds: int = 10,
    ):
        super().__init__()

        self.num_seq_text_embeds = num_seq_text_embeds
        self.seq_embeds = nn.Linear(seq_embed_dim, self.num_seq_text_embeds * cross_attention_dim)
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    def forward(self, text_embeds: torch.Tensor, seq_embeds: torch.Tensor):
        batch_size = text_embeds.shape[0]

        # seq
        seq_text_embeds = self.seq_embeds(seq_embeds)
        seq_text_embeds = seq_text_embeds.reshape(batch_size, self.num_seq_text_embeds, -1)

        # text
        text_embeds = self.text_proj(text_embeds)

        return torch.cat([seq_text_embeds, text_embeds], dim=1)


class SeqProjection(nn.Module):
    def __init__(
        self,
        seq_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_seq_text_embeds: int = 32,
    ):
        super().__init__()

        self.num_seq_text_embeds = num_seq_text_embeds
        self.seq_embeds = nn.Linear(seq_embed_dim, self.num_seq_text_embeds * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, seq_embeds: torch.Tensor):
        batch_size = seq_embeds.shape[0]

        # seq
        seq_embeds = self.seq_embeds(seq_embeds)
        seq_embeds = seq_embeds.reshape(batch_size, self.num_seq_text_embeds, -1)
        seq_embeds = self.norm(seq_embeds)
        return seq_embeds
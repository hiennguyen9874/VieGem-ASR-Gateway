"""EmbeddingGemma 2 audio encoder + CTC reader with a head in Gemma's text-embedding space.

    wav -> log-mel -> Gemma audio encoder (12 conformer layers, 25 frames/s, chunked attention)
        -> learned softmax mix of the 13 hidden states (subsampler + each layer)
        -> reader: LayerNorm + Linear + causal depthwise conv + N pre-norm transformer blocks
        -> to_text (d -> 512) @ unit_rows^T + bias, unit rows initialized from Gemma input embeddings
        -> CTC

Encoder adaptation preserves the checkpoint's frozen and top-k parameter flags.
Head: "trainable" (rows init from text embeddings and trained), "frozen" (rows fixed),
      "residual" (fixed text rows + alpha * learned delta).
Streaming: reader self-attention is restricted to the current chunk and previous chunks; the encoder is already
chunked/left-context only and the reader conv is causal.
"""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .features import LogMel


@dataclass
class ModelConfig:
    encoder_id: str = "google/embeddinggemma-2"
    vocab_size: int = 0  # incl. blank and unk
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    ff_mult: int = 4
    conv_kernel: int = 15
    layer_mix: bool = True
    encoder_mode: str = "top_k"  # frozen | top_k
    unfreeze_top_k: int = 2
    head_mode: str = "trainable"  # trainable | frozen | residual
    residual_alpha: float = 1.0
    head_cosine: bool = True  # cosine similarity with learned scale; False = plain dot product
    chunk_frames: int = 12  # 0.48 s

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_encoder_and_text_embeddings(encoder_id: str):
    full = AutoModel.from_pretrained(
        encoder_id, dtype=torch.bfloat16, vision_config=None, local_files_only=True
    )
    encoder = full.audio_tower.float()
    text_emb = full.language_model.embed_tokens.weight.detach().float().clone()  # (262144, 512)
    del full
    return encoder, text_emb


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, ff_mult):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d), nn.GELU(), nn.Identity(), nn.Linear(ff_mult * d, d)
        )

    def forward(self, x, attn_mask):
        B, T, D = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(B, T, 3, self.n_heads, D // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.norm2(x))


class CTCModel(nn.Module):
    def __init__(self, cfg: ModelConfig, gemma_ids: list[int], encoder=None, text_emb=None):
        super().__init__()
        assert cfg.vocab_size == len(gemma_ids) + 2
        self.cfg = cfg
        if encoder is None:
            encoder, text_emb = load_encoder_and_text_embeddings(cfg.encoder_id)
        self.mel = LogMel(cfg.encoder_id)
        self.encoder = encoder
        n_enc = encoder.config.num_hidden_layers
        enc_dim = encoder.config.hidden_size
        self._set_encoder_grad_flags(n_enc)

        # Collect subsampler output + every conformer layer output (output_hidden_states would give the
        # 1536-d output_proj instead of the last layer's raw output).
        self._acts = []
        grab = lambda mod, inp, out: self._acts.append(out[0] if isinstance(out, tuple) else out)  # noqa: E731
        self.encoder.subsample_conv_projection.register_forward_hook(grab)
        for layer in self.encoder.layers:
            layer.register_forward_hook(grab)

        d = cfg.d_model
        self.layer_logits = nn.Parameter(torch.zeros(n_enc + 1)) if cfg.layer_mix else None
        self.in_norm = nn.LayerNorm(enc_dim)
        self.in_proj = nn.Linear(enc_dim, d)
        self.pos_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)  # causal (left padded)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers)]
        )
        self.out_norm = nn.LayerNorm(d)

        rows = text_emb[torch.tensor(gemma_ids)]
        rows = rows / rows.norm(dim=-1, keepdim=True).mean()  # put rows on unit average scale
        text_dim = rows.shape[1]
        self.to_text = nn.Linear(d, text_dim)
        if cfg.head_mode == "trainable":
            self.unit_rows = nn.Parameter(rows.clone())
        elif cfg.head_mode == "frozen":
            self.register_buffer("unit_rows", rows.clone())
        elif cfg.head_mode == "residual":
            self.register_buffer("text_rows", rows.clone())
            self.delta_rows = nn.Parameter(torch.zeros_like(rows))
        else:
            raise ValueError(cfg.head_mode)
        self.special_rows = nn.Parameter(torch.randn(2, text_dim) * 0.02)  # blank, unk
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        if cfg.head_cosine:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def _set_encoder_grad_flags(self, n_enc):
        cfg = self.cfg
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if cfg.encoder_mode == "top_k":
            for layer in self.encoder.layers[n_enc - cfg.unfreeze_top_k :]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        elif cfg.encoder_mode != "frozen":
            raise ValueError(cfg.encoder_mode)

    def head_rows(self):
        if self.cfg.head_mode == "residual":
            rows = self.text_rows + self.cfg.residual_alpha * self.delta_rows
        else:
            rows = self.unit_rows
        return torch.cat([self.special_rows, rows], dim=0)

    @staticmethod
    def chunk_mask(T, chunk, left_chunks, device):
        c = torch.arange(T, device=device) // chunk
        allow = c[None, :] <= c[:, None]
        if left_chunks is not None:
            allow &= c[None, :] >= c[:, None] - left_chunks
        return allow

    def forward(self, wav, wav_lens, streaming=False, left_chunks=None):
        """Returns float32 logits (B, T, V) and output lengths (B,)."""
        feats, fmask = self.mel(wav, wav_lens)
        self._acts.clear()
        out = self.encoder(feats, fmask, return_dict=True)
        acts, self._acts = self._acts, []
        mask = out.attention_mask
        if self.cfg.layer_mix:
            w = torch.softmax(self.layer_logits.float(), dim=0)
            h = sum(w[i] * a.float() for i, a in enumerate(acts))
        else:
            h = acts[-1].float()

        x = self.in_proj(self.in_norm(h))
        xc = F.pad((x * mask[..., None]).transpose(1, 2), (self.cfg.conv_kernel - 1, 0))
        x = x + F.gelu(self.pos_conv(xc).transpose(1, 2))

        T = x.shape[1]
        attn_mask = mask[:, None, None, :]
        chunk = None
        if streaming:
            chunk = self.cfg.chunk_frames
        if chunk is not None:
            attn_mask = attn_mask & self.chunk_mask(T, chunk, left_chunks, x.device)[None, None]
        # Rows that are fully padded would attend to nothing -> NaN; let them see themselves.
        attn_mask = attn_mask | torch.eye(T, dtype=torch.bool, device=x.device)[None, None]
        for blk in self.blocks:
            x = blk(x, attn_mask)

        t = self.to_text(self.out_norm(x)).float()
        rows = self.head_rows().float()
        if self.cfg.head_cosine:
            logits = self.logit_scale.exp() * (
                F.normalize(t, dim=-1) @ F.normalize(rows, dim=-1).t()
            )
        else:
            logits = t @ rows.t()
        logits = logits + self.head_bias.float()
        return logits, mask.sum(1)


def greedy_decode(logits, lens, blank=0):
    best = logits.argmax(-1).cpu()
    out = []
    for b in range(best.shape[0]):
        seq = best[b, : int(lens[b])]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != blank].tolist())
    return out

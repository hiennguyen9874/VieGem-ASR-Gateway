"""Batched GPU log-mel matching transformers' Gemma4AudioFeatureExtractor."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor


class LogMel(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        assert fe.preemphasis == 0.0 and fe.dither == 0.0 and fe.input_scale_factor == 1.0
        assert fe.per_bin_mean is None and fe.per_bin_stddev is None
        self.frame_length, self.hop_length, self.fft_length = (
            fe.frame_length,
            fe.hop_length,
            fe.fft_length,
        )
        self.mel_floor = float(fe.mel_floor)
        self.register_buffer(
            "window", torch.tensor(fe.window, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False
        )

    @torch.no_grad()
    def forward(
        self, wav: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """wav (B, L) float32 zero-padded -> feats (B, T, 128), mask (B, T)."""
        pad_left = self.frame_length // 2  # semicausal padding, as in the HF extractor
        wav = F.pad(wav.float(), (pad_left, 0))
        frames = wav.unfold(-1, self.frame_length + 1, self.hop_length)[..., :-1]
        spec = torch.fft.rfft(frames * self.window, n=self.fft_length, dim=-1).abs()
        feats = torch.log(spec @ self.mel_filters + self.mel_floor)
        # A frame is valid when its whole analysis window is real audio.
        frame_end = (
            torch.arange(feats.shape[1], device=wav.device) * self.hop_length + self.frame_length
        )
        mask = frame_end[None, :] < (wav_lens[:, None] + pad_left)
        return feats * mask[..., None], mask

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Various utilities."""

from hashlib import sha256
from pathlib import Path
import typing as tp

import torch
import torchaudio


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _get_checkpoint_url(root_url: str, checkpoint: str):
    if not root_url.endswith('/'):
        root_url += '/'
    return root_url + checkpoint


def _check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, 'rb') as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum:
        raise RuntimeError(f'Invalid checksum for file {path}, '
                           f'expected {checksum} but got {actual_checksum}')


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


class EMA:
    def __init__(self, models, decay=0.99):
        """
        初始化 EMA 类，支持多个子模块。
        
        参数:
        - models (list of torch.nn.Module): 需要应用 EMA 的子模块列表。
        - decay (float): EMA 的衰减因子。
        """
        self.decay = decay
        self.shadow = {}

        # 为每个子模块初始化 shadow
        for model in models:
            device = next(model.parameters()).device
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[f"{model}.{name}"] = param.clone().detach().to(device)

    def update(self, models):
        """
        更新所有子模块的 EMA 参数。
        """
        for model in models:
            device = next(model.parameters()).device
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[f"{model}.{name}"] = (
                        self.decay * self.shadow[f"{model}.{name}"].to(device) + 
                        (1.0 - self.decay) * param.detach()
                    )

    def apply_shadow(self, models):
        """
        将所有子模块的 EMA 参数应用到模型上。
        """
        for model in models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[f"{model}.{name}"])

    def store(self, models):
        """
        存储所有子模块的当前权重，用于恢复。
        """
        self.backup = {}
        for model in models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[f"{model}.{name}"] = param.clone()

    def restore(self, models):
        """
        恢复所有子模块的原始权重。
        """
        for model in models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[f"{model}.{name}"])

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelScale, Spectrogram

class LogMelSpec(torch.nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 #win_ms=25, hop_ms=10,
                 win=400, hop=160,
                 n_freq=201, n_mels=40,
                 **kwargs):

        super(LogMelSpec, self).__init__()
        self.eps = 1e-10 

        #win = round(win_ms * sample_rate / 1000)
        #hop = round(hop_ms * sample_rate / 1000)
        n_fft = (n_freq - 1) * 2
        self._win_args = {"n_fft": n_fft, "hop_length": hop, "win_length": win}
        self.register_buffer("_window", torch.hann_window(win))

        self._stft_args = {
            "center": False,#True,
            "pad_mode": "reflect",
            "normalized": False,
            "onesided": True,
        }
        # stft_args: same default values as torchaudio.transforms.Spectrogram & librosa.core.spectrum._spectrogram
        self._stft = partial(torch.stft, **self._win_args, **self._stft_args)
        self._melscale = MelScale(sample_rate=sample_rate, n_mels=n_mels)
        self._istft = partial(torch.istft, **self._win_args, **self._stft_args)

    def _magphase(self, spectrogram):
        # Replace the deprecated private member function self._magphase
        # Ref. https://github.com/pytorch/audio/issues/1337
        # original: `self._magphase = partial(torchaudio.functional.magphase, power=2)`
        spectrogram = torch.view_as_complex(spectrogram)
        return spectrogram.abs().pow(exponent=2), spectrogram.angle()

    def forward(self, wavs):
        complx = self._stft(wavs, window=self._window)
        linear, phase = self._magphase(complx)
        mel = self._melscale(linear)
        return (mel + self.eps).log()

    def istft(self, linears=None, phases=None, linear_power=2, complxs=None):
        assert complxs is not None or (linears is not None and phases is not None)
        # complxs: (*, n_freq, max_feat_len, 2) or (*, max_feat_len, n_freq * 2)
        # linears, phases: (*, max_feat_len, n_freq)

        if complxs is None:
            linears, phases = self._transpose_list([linears, phases])
            complxs = linears.pow(1 / linear_power).unsqueeze(-1) * torch.stack(
                [phases.cos(), phases.sin()], dim=-1
            )
        if complxs.size(-1) != 2:
            # treat complxs as: (*, max_feat_len, n_freq * 2)
            shape = complxs.size()
            complxs = complxs.view(*shape[:-1], -1, 2).transpose(-2, -3).contiguous()
        # complxs: (*, n_freq, max_feat_len, 2)

        return self._istft(complxs, window=self._window)
        # return: (*, max_wav_len)

def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True

    return mask

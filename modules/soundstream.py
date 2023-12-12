# copied from soundstream==0.0.1 
from functools import reduce
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from vector_quantize_pytorch import ResidualVQ

def _infer_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_codec(device):
    #device = _infer_device()
    checkpoint_path = hf_hub_download(
        repo_id='haydenshively/SoundStream', 
        filename='soundstream_variant_naturalspeech2.pt')

    model_naturalspeech2 = SoundStream(
        n_q=16,
        codebook_size=1024,
        D=256,
        C=58,
        strides=(2, 4, 5, 5),
    )
    model_naturalspeech2.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model_naturalspeech2.eval()

    for p in model_naturalspeech2.parameters():
        p.requires_grad_(False)

    return model_naturalspeech2

class SoundStream(nn.Module):
    def __init__(self, n_q, codebook_size, D, C, strides=(2, 4, 5, 8)):
        super(SoundStream, self).__init__()

        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(C=C, D=D, strides=strides)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q,
            codebook_size=codebook_size,
            dim=D,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D, strides=strides)

    def forward(
            self,
            x,
            mode: Literal['end-to-end', 'encode', 'decode', 'quantize'] = 'end-to-end',
        ):
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)

        if mode == 'end-to-end':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            o = self.decoder(quantized.permute((0,2,1)))
            return o
        
        if mode == 'encode':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            return quantized
        
        if mode == 'decode':
            o = self.decoder(x.permute((0,2,1)))
            return o

        if mode == 'quantize':
            e = self.encoder(x)
            _, indices, _ = self.quantizer(e.permute((0,2,1)))
            return indices

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(EncoderBlock, self).__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=9
            ),
            CausalConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            EncoderBlock(out_channels=2*C, stride=strides[0]),
            EncoderBlock(out_channels=4*C, stride=strides[1]),
            EncoderBlock(out_channels=8*C, stride=strides[2]),
            EncoderBlock(out_channels=16*C, stride=strides[3]),
            CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(DecoderBlock, self).__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=16*C, kernel_size=7),
            DecoderBlock(out_channels=8*C, stride=strides[3]),
            DecoderBlock(out_channels=4*C, stride=strides[2]),
            DecoderBlock(out_channels=2*C, stride=strides[1]),
            DecoderBlock(out_channels=C, stride=strides[0]),
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        )

    def forward(self, x):
        return self.layers(x)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode='reflect', **kwargs):
        super(CausalConv1d, self).__init__()

        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)

        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, [self.causal_padding, 0], mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(CausalConvTranspose1d, self).__init__()

        self.upsample_factor = stride
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            **kwargs
        )

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        # https://github.com/lucidrains/audiolm-pytorch/issues/8
        return out[..., :(n * self.upsample_factor)]


class ResidualUnit(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            pad_mode='reflect'
        ):
        super(ResidualUnit, self).__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                pad_mode=pad_mode,
            ),
            nn.ELU(),
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                pad_mode=pad_mode,
            ),
            nn.ELU(),
        )

    def forward(self, x):
        return x + self.layers(x)

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pad_to_multiple, compute_mask_indices, LogMelSpec
from utils.utils import accuracy
from .soundstream import load_codec
from .multihead_attention import MultiheadAttention

class Selectra(nn.Module):
    def __init__(self, 
                 emb = 512, enc_emb = 768, enc_layers = 6, #12,
                 #mask_prob = 0.65, mask_length = 10):
                 mask_prob = 0.1, mask_length = 5, dev = 'cuda:0'):

        super().__init__()
        self.emb = emb
        self.enc_emb = enc_emb

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(self.enc_emb).uniform_())

        self.pre_conv = nn.Sequential(nn.Conv1d(1, 32, kernel_size=1), nn.GELU())
        #feat_enc_layers = [(512, 10, 5)] + \
        #        [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
        feat_enc_layers = [(64, 10, 5)] * 2 + [(128, 8, 4), (192, 4, 2)]
        #feat_enc_layers = [(64, 4, 2), (128, 8, 4)] + [(256, 10, 5)] * 2
        self.feat_enc = FeatureEncoder(32, self.enc_emb,
            conv_layers=feat_enc_layers, dropout=0.0, conv_bias=False)

        self.codec = load_codec(dev)
        self.num_quant, self.quant_emb, _ = self.codec.quantizer.codebooks.shape

        self.generator = TransformerEncoder(
            enc_layers, self.enc_emb, self.enc_emb // 4, 3072 // 4,
            dropout = 0., layerdrop = 0.)
        self.gen_projs = [nn.Linear(self.enc_emb // 4, \
            self.quant_emb, bias = False, device=dev) \
            for _ in range(self.num_quant)]

        self.discriminator = TransformerEncoder(
            enc_layers, self.enc_emb, self.enc_emb, 3072,
            dropout = 0., layerdrop = 0.)
        self.disc_proj = nn.Linear(self.enc_emb, 2)

    def forward(self, x, padding_mask = None, mask = False):
        self.codec.eval()

        x_orig = x
        x = x.unsqueeze(1)

        if not mask:

            x_disc = self.feat_enc(self.pre_conv(x))
            x_disc = self.discriminator(x_disc)

            return x_disc

        x_q = self.codec(x, mode='quantize').detach()
        x = self.feat_enc(self.pre_conv(x))

        B, T, C = x.shape
        mask_indices = compute_mask_indices(
            (B, T), padding_mask,
            self.mask_prob, self.mask_length,
            'static', 0.,
            min_masks=2, no_overlap=False, min_space=1,
            require_same_masks=True, mask_dropout=0.
        )
        mask_indices = torch.from_numpy(mask_indices).to(x.device)
        x[mask_indices] = self.mask_emb 

        x = self.generator(x)

        x_projs = []
        x_indices = []

        x_proj = self.gen_projs[0](x)
        uniform_ns = torch.rand(x_proj.shape)
        for i in range(len(self.gen_projs)):
            x_proj = self.gen_projs[i](x)
            x_projs.append(x_proj.transpose(2, 1))

            # uniform_ns = torch.rand(x_proj.shape)
            gumbel_ns = -torch.log(-torch.log(uniform_ns + 1e-9) + 1e-9)
            logits = F.softmax(x_proj + gumbel_ns.to(x.device), -1)
            indices = torch.argmax(logits, -1)
            x_indices.append(indices)

        x_projs = torch.stack(x_projs, -1)
        x_indices = torch.stack(x_indices, -1)
        # TODO temporary fix; need to add padding scheme same as soundstream
        if x_q.shape[1] > x_indices.shape[1]:
            x_q = x_q[:,:x_indices.shape[1],:]

        out_acc = accuracy(x_indices[:,:,0], x_q[:,:,0], 'libri')
        mlm_loss = F.cross_entropy(x_projs, x_q, reduction='none')
        mlm_loss = (mask_indices.unsqueeze(-1) * mlm_loss).sum()
        mlm_loss /= (B * T)
        # mlm_loss = F.cross_entropy(x_projs[:,:,:,0], x_q[:,:,0], reduction='none')
        # mlm_loss = (mask_indices * mlm_loss).sum(1) / mask_indices.sum(1)
        # mlm_loss = mlm_loss.mean()

        x_gen = torch.zeros_like(x_q)
        x_gen[~mask_indices] += x_q[~mask_indices]
        x_gen[mask_indices] += x_indices[mask_indices]

        x_gen = self.codec.quantizer.get_output_from_indices(x_gen)
        x_gen = self.codec(x_gen, mode='decode')
        x_gen_pad = max(x_orig.shape[-1] - x_gen.shape[-1], 0)
        x_gen = F.pad(x_gen, (0, x_gen_pad))

        x_disc = self.feat_enc(self.pre_conv(x_gen))
        x_disc = self.discriminator(x_disc)
        x_disc = self.disc_proj(x_disc)

        disc_loss = F.cross_entropy(x_disc.transpose(2,1), mask_indices.long())

        return mlm_loss, disc_loss, out_acc

class FeatureEncoder(nn.Module):
    def __init__(self, in_d, out_d,
                 conv_layers: List[Tuple[int, int, int]],
                 dropout: float = 0.0,
                 conv_bias: bool = False):

        super().__init__()
        self.conv_layers = nn.ModuleList()

        for i, cl in enumerate(conv_layers):
            (dim, k, stride) = cl
            
            conv = nn.Conv1d(in_d, dim, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            convs = nn.Sequential(conv, nn.Dropout(p=dropout), nn.GELU())

            self.conv_layers.append(convs)
            in_d = dim

        self.layer_norm = nn.LayerNorm(in_d)
        self.post_proj = nn.Linear(in_d, out_d)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = self.post_proj(x)

        return x

def make_conv_pos(e, k, g):
    pad = nn.ConstantPad1d(((k//2)-1, k//2), 0)
    pos_conv = nn.Conv1d(
        e, e, kernel_size=k, groups=g)

    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pad, pos_conv, nn.GELU())

    return pos_conv

class TransformerEncoder(nn.Module):
    def __init__(self, enc_layers = 12, 
                 in_emb = 768, emb = 768, fft_emb = 3072, 
                 dropout: float = 0.1,
                 layerdrop: float = 0.05):

        super().__init__()

        self.dropout = dropout
        self.in_emb = in_emb
        self.emb = emb
        self.enc_layers = enc_layers

        self.pos_conv = make_conv_pos(self.in_emb, 128, 16)

        self.emb_conv = None
        if self.in_emb != self.emb:
            self.emb_conv = nn.Linear(self.in_emb, self.emb)

        self.layers = nn.ModuleList(
            [TransformerEncLayer(self.emb, fft_emb) for _ in range(self.enc_layers)]
        )
        self.layer_norm = nn.LayerNorm(self.in_emb)
        self.layerdrop = layerdrop

        #self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)

        x = x + x_conv
        
        x = self.layer_norm(x)

        if self.emb_conv is not None:
            x = self.emb_conv(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            dropout_prob = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_prob > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )

        x = x.transpose(0, 1)
        return x

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

class TransformerEncLayer(nn.Module):
    def __init__(self,
                 emb: float = 768, ffn_emb: float = 3072,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation_dropout: float = 0.1) -> None:

        super().__init__()
        # Initialize parameters
        self.emb = emb
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.self_attn = MultiheadAttention(
            self.emb,
            num_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.emb)
        self.fc1 = nn.Linear(self.emb, ffn_emb)
        self.fc2 = nn.Linear(ffn_emb, self.emb)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.emb)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        residual = x

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            need_weights=False,
        )

        x = self.dropout1(x)
        x = residual + x

        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        layer_result = x

        x = self.dropout3(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x, (attn, layer_result)

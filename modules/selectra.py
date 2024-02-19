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
from einops import reduce
import random
import pdb

class Selectra(nn.Module):
    def __init__(self, 
                 emb = 512, enc_emb = 768, enc_layers = 12, #12,
                 #mask_prob = 0.65, mask_length = 10):
                 mask_prob = 0.65, mask_length = 10, dev = 'cuda:0'):

        super().__init__()
        self.emb = emb
        self.enc_emb = enc_emb

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(self.enc_emb).uniform_())

        self.codec = load_codec(dev)
        self.num_quant, self.quant_emb, self.quant_dim = self.codec.quantizer.codebooks.shape
        
        self.selected_num_quant = 16 # 사용할 rvq 개수 (1~16)
        self.post_proj = nn.Linear(self.selected_num_quant, self.enc_emb)
        self.selectra = TransformerEncoder(
            enc_layers, self.enc_emb, self.enc_emb, self.enc_emb*4,
            dropout = 0., layerdrop = 0.)
        
        # self.gen_split = nn.Linear(1, self.selected_num_quant)
        # self.gen_projs = [nn.Linear(self.enc_emb, \
        #     self.quant_emb, bias = False, device=dev) \
        #     for _ in range(self.selected_num_quant)]
        self.gen_projs_split = nn.Linear(1, self.selected_num_quant)
        self.gen_projs = nn.Linear(self.enc_emb, self.quant_emb)
        self.disc_proj = nn.Linear(self.enc_emb, 2)
        
        # self.discriminator = TransformerEncoder(
        #     enc_layers, self.enc_emb, self.enc_emb, self.enc_emb*4,
        #     dropout = 0., layerdrop = 0.)

        self.disc_proj_1 = nn.Linear(self.enc_emb, self.enc_emb)
        self.disc_proj_2 = nn.Linear(self.enc_emb, self.selected_num_quant)
        self.disc_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, wav, padding_mask = None, mask = False):
        self.codec.eval()

        wav = wav.unsqueeze(1)

        x = self.codec(wav, mode='quantize').detach().permute(0,2,1) #B,256,T
        x_q = self.codec(wav, mode='quantize').detach() #B,T,n_q
        # x_q_summed = self.codec(wav, mode='encode').detach()
        
        # x_q = reduce(x_q, 'q ... -> ...', 'sum') #(2, 978, 256)
        if not mask:
            x_disc = self.selectra(x_q)
            return x_disc

        x = self.post_proj(x.permute(0,2,1).float())
        B, T, C = x.shape
        # B,n_q,T = x.shape
        mask_indices = compute_mask_indices(
            (B, T), padding_mask,
            self.mask_prob, self.mask_length,
            'static', 0.,
            min_masks=2, no_overlap=False, min_space=1,
            require_same_masks=True, mask_dropout=0.
        )
        mask_indices = torch.from_numpy(mask_indices).to(x.device)
        
        x[mask_indices] = self.mask_emb
        x = self.selectra(x) #B,T,256

        # x_projs = []
        x_indices = []
        
        # x_split = self.gen_split(x.unsqueeze(-1)).permute(3,0,1,2)
        # x_split_sum_hat = torch.sum(x_split, dim=0)
        
        x_proj = self.gen_projs_split(x.unsqueeze(-1)).permute(0,1,3,2) #B,T,n_q,256
        x_proj = self.gen_projs(x_proj) # B,T,n_q,1024
        for i in range(self.selected_num_quant):
            # x_proj = self.gen_projs[i](x) #(2, 978, 1024)
            # x_projs.append(x_proj.permute(2,0,1,3)[i]) # input n_q,B,T,1024 -> B,T,1024
            x_proj_temp = x_proj.permute(2,0,1,3)[i]
            uniform_ns = torch.rand(x_proj_temp.shape)
            gumbel_ns = -torch.log(-torch.log(uniform_ns + 1e-9) + 1e-9)
            logits = F.softmax(x_proj_temp + gumbel_ns.to(x.device), -1)
            indices = torch.argmax(logits, -1)
            x_indices.append(indices)
        # x_projs = torch.stack(x_projs, -1) #(2, 1024, 978, 6) B,1024,T,n_q
        x_projs = x_proj.permute(0,3,1,2) # B,1024,T,n_q
        # x_origin : (6, 2, 978, 256)
        # x_q : #(2, 978, 6)
        x_indices = torch.stack(x_indices, -1) #(2, 978, 6) B,T,n_q
        # TODO temporary fix; need to add padding scheme same as soundstream
        if x_q.shape[1] > x_indices.shape[1]:
            x_q = x_q[:,:x_indices.shape[1],:]
        out_acc = accuracy(x_indices, x_q, 'libri')
        # [B,T,n_q]끼리 비교후 해당 frame의 n_q가 모두 같을 떄 True로 disc_label 구하기
        disc_label = x_indices == x_q
        # disc_label = torch.all(disc_label, dim=-1)
        disc_label = disc_label.int()

        mlm_loss = F.cross_entropy(x_projs, x_q.long(), reduction='none') #(B, T, n_q)
        ##################### MSE #####################
        # n_q_l1_loss = F.l1_loss(x_q_summed, x_split_sum_hat)
        # print(mask_indices.unsqueeze(-1).shape)
        # (2, 978, 1)
        mlm_loss = (mask_indices.unsqueeze(-1) * mlm_loss).sum()
        mlm_loss /= mask_indices.sum()

        x_disc = self.selectra(x)
        # x_disc = self.discriminator(x_in)
        
        # disc_loss = F.cross_entropy(x_disc.transpose(2,1), mask_indices.long())
        
        x_disc = F.relu(self.disc_proj_1(x_disc))
        x_disc = self.disc_proj_2(x_disc)#.squeeze(-1)
        x_disc = F.sigmoid(x_disc)
        disc_loss = self.disc_loss_fn(x_disc, disc_label.float())
        # self.discriminator = TransformerEncoder(
        #     enc_layers, self.enc_emb, self.enc_emb, 3072,
        #     dropout = 0., layerdrop = 0.)
  
        return mlm_loss, disc_loss, out_acc#, n_q_l1_loss
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

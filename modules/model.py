import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths
import random
from .selectra import Selectra

class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        
        self.hidden_dim = self.hp['model']['hidden_dim']
        self.enc_hidden_dim = self.hp['model']['enc_hidden_dim']
        self.model = Selectra(self.hidden_dim, self.enc_hidden_dim)

        self.nclass = self.hp['model']['n_symbols']
        self.fc     = nn.Linear(self.enc_hidden_dim, self.nclass)
        
    def outputs(self, wav_padded):
        feat = self.model(wav_padded, mask = True)
        logits = self.fc(feat)
        
        return logits
    
    def forward(self, wav_padded, wav_lengths, text_padded, text_lengths, criterion):
        #import pdb
        #pdb.set_trace()
        outputs     = self.outputs(wav_padded)
        wav_lengths = wav_lengths // 320
        ctc_loss    = criterion(outputs.transpose(1,0), 
                text_padded, wav_lengths, text_lengths)

        return ctc_loss, outputs
        

import torch
import torch.nn as nn
from .selectra import Selectra
from utils.utils import *

class Model(nn.Module):
    def __init__(self, hp, device):
        super(Model, self).__init__()
        self.hp = hp
        self.device = device
        self.hidden_dim = self.hp['model']['hidden_dim']
        self.enc_hidden_dim = self.hp['model']['enc_hidden_dim']
        self.model = Selectra(self.hidden_dim, self.enc_hidden_dim, dev = self.device)
        self.nclass = self.hp['model']['n_symbols']
        self.fc = nn.Linear(self.enc_hidden_dim, self.nclass)
        self.fc_sv  = nn.Linear(self.enc_hidden_dim, 1251)
        self.fc_ks  = nn.Linear(self.enc_hidden_dim, 30)
        
    def forward(self, wav_padded, label_padded, wav_lengths=None, label_lengths=None, criterion=None, mask=True, data_name=None):
        if mask:
            return self.model(wav_padded, mask = mask)
        else:
            """
            asr: label_padded: script
            sv:  label_padded: speaker id
            sv:  label_padded: keyword id
            """

            x_disc      = self.model(wav_padded, mask = mask)
            if data_name == 'libri':
                logits      = self.fc(x_disc) # B, T, C
                logits      = logits.transpose(0,1).log_softmax(2)
                wav_lengths = wav_lengths // 200 - 2
                ctc_loss    = criterion(logits, label_padded, wav_lengths, label_lengths)
                return ctc_loss
            elif data_name =='vox1':
                spkemb      = torch.mean(x_disc, 1) 
                logits      = self.fc_sv(spkemb) # B, T, C
                ind_estimated = torch.argmax(logits, 1)
                cls_loss    = criterion(logits, label_padded)
                acc_out     = accuracy(ind_estimated, label_padded, data_name) 
                return cls_loss, acc_out
            elif data_name == 'KeywordSpotting':
                spkemb      = torch.mean(x_disc, 1) 
                logits      = self.fc_ks(spkemb) # B, T, C
                ind_estimated = torch.argmax(logits, 1)
                cls_loss    = criterion(logits, label_padded)
                acc_out     = accuracy(ind_estimated, label_padded, data_name) 
                return cls_loss, acc_out
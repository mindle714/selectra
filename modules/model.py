import torch
import torch.nn as nn
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
        
    def forward(self, wav_padded, wav_lengths, text_padded, text_lengths, criterion):
        ctc_loss = self.model(wav_padded, mask = True)
        return ctc_loss
        

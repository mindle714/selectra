import torch
import torch.nn as nn
from .selectra import Selectra

class Model(nn.Module):
    def __init__(self, hp, device):
        super(Model, self).__init__()
        self.hp = hp
        self.device = device
        self.hidden_dim = self.hp['model']['hidden_dim']
        self.enc_hidden_dim = self.hp['model']['enc_hidden_dim']
        self.model = Selectra(self.hidden_dim, self.enc_hidden_dim, dev = self.device)

        self.nclass = self.hp['model']['n_symbols']
        self.fc     = nn.Linear(self.enc_hidden_dim, self.nclass)
        
    def forward(self, wav_padded, wav_lengths, text_padded, text_lengths, criterion, mask=True):
        if mask:
            return self.model(wav_padded, mask = mask)
        else:
            #self.model.eval()
            #with torch.no_grad():
            x_disc  = self.model(wav_padded, mask = mask)
            logits      = self.fc(x_disc) # B, T, C
            logits      = logits.transpose(0,1).log_softmax(2).detach().requires_grad_()
            wav_lengths = wav_lengths // 200 - 2
            ctc_loss    = criterion(logits, text_padded, wav_lengths, text_lengths)
            return ctc_loss
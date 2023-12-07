import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths
import random
import fairseq

class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        
        cp_path = 'wav2vec_small.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.hidden_dim = self.hp['model']['hidden_dim']
        self.nclass = self.hp['model']['n_symbols']
        self.fc     = nn.Linear(512, self.nclass)
        
    def outputs(self, wav_padded):

        self.model.eval()
        with torch.no_grad():
            wav = self.model.feature_extractor(wav_padded)
        logits = self.fc(wav.transpose(1,2))
        
        return logits
    
    def forward(self, wav_padded, wav_lengths, text_padded, text_lengths, criterion):
        import pdb
        pdb.set_trace()
        outputs     = self.outputs(wav_padded)
        wav_lengths = wav_lengths // 320
        ctc_loss    = criterion(outputs, text_padded, wav_lengths, text_lengths)

        return ctc_loss, outputs
        
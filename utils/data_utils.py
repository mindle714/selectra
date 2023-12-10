import random
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import pickle as pkl
import json
import yaml
import random
from text import text_to_sequence
import soundfile as sf

def load_filepaths(path_ema):
    with open(path_ema, 'r') as f:
        lines = f.readlines()
        filepaths = [line.strip('\n') for line in lines]
    return filepaths

class AudioSet(torch.utils.data.Dataset):
    def __init__(self, process_type, hparams):
        random.seed(hparams['train']['seed'])
        self.process_type = process_type
        if process_type == 'train':
            self.list_wavs = load_filepaths(hparams['train']['training_files'])
            random.shuffle(self.list_wavs)
        elif process_type == 'val':
            self.list_wavs = load_filepaths(hparams['train']['validation_files'])
        else:
            raise Exception('Choose between [train, val]')

        self.sr_wav = hparams['train']['validation_files']
        self.data_path = hparams['train']['data_path']

    def get_data_pair(self, path_data):

        path_wav, script = path_data.split('|')
        path_wav = os.path.join(self.data_path, path_wav)

        wav, _ = sf.read(path_wav)
        wav = wav[:((len(wav)//320) -1)* 320 + 80]
        #print(wav.shape[0] / 320, wav.shape[0])

        script_id = text_to_sequence(script, ['custom_english_cleaners'])
        #print(wav.shape, len(script_id))
        wav = torch.FloatTensor(wav)
        script_id = torch.LongTensor(script_id)

        return wav, script_id, script

    def __getitem__(self, index):
        return self.get_data_pair(self.list_wavs[index])

    def __len__(self):
        return len(self.list_wavs)


class AudioSetCollate():
    def __init__(self, eval=False):
        self.eval = eval
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length

        wav_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[0] for x in batch]),
            dim=0, descending=True)

        max_input_len = wav_lengths[0]
        wav_padded    = torch.zeros(len(batch), max_input_len)

        # wav
        max_txt_len = max([len(x[1]) for x in batch])
        txt_padded  = torch.zeros(len(batch), max_txt_len)

        txt_lengths = torch.zeros(len(batch), dtype=int)
        for i in range(len(ids_sorted_decreasing)):

            wav = batch[ids_sorted_decreasing[i]][0]
            wav_padded[i, :wav.shape[0]] = wav

            txt = batch[ids_sorted_decreasing[i]][1] 
            txt_padded[i, :txt.shape[0]] = txt
            txt_lengths[i] = txt.shape[0]

        return wav_padded, wav_lengths, txt_padded, txt_lengths



if __name__ == "__main__":

    config_path = '/home/miseul/cousework/it_hgkang/selectra/configs/default.yaml'
    with open(config_path) as fp:
        config = yaml.full_load(fp)
    print('Test')
    trainset = AudioSet('train', config)
    collate_fn   = AudioSetCollate()
    train_loader = DataLoader(trainset,
                            shuffle=True,
                            batch_size=config['train']['batch_size'], 
                            collate_fn= collate_fn,
                            drop_last=True)

    import tqdm
    for wav_padded, wav_lengths, txt_padded, txt_lengths in tqdm.tqdm(train_loader):
        #print(wav_padded.shape)
        pass

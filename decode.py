import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import Model
import yaml
from utils.data_utils import *
from utils.writer import get_writer
#from jiwer import cer
from utils.utils import *
import tqdm
import pdb

# 문자를 숫자로 매핑하는 딕셔너리
char_to_int = {'ctc blank': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
int_to_char = {i: ch for ch, i in char_to_int.items()}

def decode(model, criterion, val_loader, iteration, writer, device, data_name):
    
    model.eval()
    all_predictions = []
    all_ground_truths = []
    with torch.no_grad():

        n_data, val_loss = 0, 0
        for i, batch in enumerate(tqdm.tqdm(val_loader)):

            n_data += len(batch[0])
            wav_padded, wav_lengths, txt_padded, txt_lengths = [
                x.to(device) for x in batch
            ]

            ctc_loss,logits  = model(wav_padded, 
                              txt_padded, 
                              wav_lengths, 
                              txt_lengths, 
                              criterion, 
                              mask=False,
                              data_name=data_name)
            # val_loss += ctc_loss.item() * len(batch[0])

            
            argmax_outputs = torch.argmax(logits,dim=2)
            indices = torch.unique_consecutive(argmax_outputs, dim=0)
            decoded_outputs = greedy_decode_ctc(indices)
            # pdb.set_trace()

            txt_padded = txt_padded.squeeze().detach().cpu().numpy()
            hypothesis = [''.join([int_to_char[idx] for idx in decoded_outputs])]
            print("predict:", hypothesis)
            t_hypothesis = [''.join([int_to_char[idx] for idx in txt_padded])]
            print("gt:", t_hypothesis)


import torch
import numpy as np

def greedy_decode_ctc(outputs, blank_label=0):
    sequence = []
    last_elem = None
    for elem in outputs:
        if elem != last_elem and elem != blank_label:
            sequence.append(elem.item())
        last_elem = elem

    return sequence

def main(args):

    config_path = args.c
    with open(config_path) as fp:
        config = yaml.full_load(fp)

    train_steps  = config['optimization']['train_steps']
    accumulation = config['optimization']['accumulation']
    iters_per_checkpoint = config['optimization']['iters_per_checkpoint']
    grad_clip_thresh     = config['optimization']['grad_clip_thresh']
    lr = config['optimization']['lr']
    iters_per_validation = config['optimization']['iters_per_validation']
    output_directory = config['train']['output_directory']
    pretrained_name  = config['train']['output_name']
    output_name      = f'{pretrained_name}_downstream'
    selectra_checkpoint = config['train']['selectra_checkpoint']
    data_name = config['train']['training_files'].split('/')[-1].split('_')[0]

    device   = torch.device(f'cuda:{str(args.gpu)}')

    trainset = AudioSet('train', config)
    collate_fn   = AudioSetCollate()
    train_loader = DataLoader(trainset,
                            shuffle=True,
                            batch_size=config['train']['batch_size'], 
                            collate_fn= collate_fn,
                            drop_last=True)

    valset = AudioSet('val', config)
    collate_fn = AudioSetCollate()
    val_loader = DataLoader(valset,
                            shuffle=True,
                            batch_size=1, 
                            collate_fn=collate_fn,
                            drop_last=True)

    model     = Model(config, f'cuda:{str(args.gpu)}').to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)

    criterion = nn.CTCLoss(blank=0)
    writer    = get_writer(output_directory, output_name)
    copy_file(config_path, os.path.join(output_directory, output_name, config_path.split('/')[-1]))
    loss = 0
    iteration = 0

    ### Load pre-trained model ###
    load_checkpoint(model, optimizer, selectra_checkpoint, f'{output_directory}/{pretrained_name}', device)

    decode(model, criterion, val_loader, iteration, writer, device, data_name)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--c', type=str, default='configs/default.yaml')
    p.add_argument('--iteration', type=int, default=None)

    args = p.parse_args()
    
    config_path = args.c
    with open(config_path) as fp:
        config = yaml.full_load(fp)

    os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
    torch.manual_seed(config['train']['seed'])
    torch.cuda.manual_seed(config['train']['seed'])
        
    main(args)
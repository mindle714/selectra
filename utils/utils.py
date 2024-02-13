from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import yaml
import shutil
from utils.data_utils import *
import torch.nn as nn

def accuracy(ind_estimated, ind_target, data_name):
    count   = torch.sum(ind_estimated == ind_target)
    if data_name == 'libri':
        results = (count / (ind_estimated.shape[0] * ind_estimated.shape[1] * ind_estimated.shape[2])) * 100
    elif data_name == 'vox1':
        results = count / ind_estimated.shape[0] * 100
    elif data_name == 'KeywordSpotting':
        results = count / ind_estimated.shape[0]* 100
    return results

def load_yaml(path):
    with open(path, 'r') as f:
        out = f.read()
    out = yaml.safe_load(out)

    return out

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):

    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")

def load_checkpoint(model, optimizer, iteration, filepath, device):

    checkpoint = torch.load(f'{filepath}/checkpoint_{iteration}', map_location=f'cuda:{device.index}')
    checkpoint['state_dict']['fc_sv.weight'] = torch.FloatTensor(torch.empty(1251, 256))
    nn.init.trunc_normal_(checkpoint['state_dict']['fc_sv.weight'])

    checkpoint['state_dict']['fc_ks.weight'] = torch.FloatTensor(torch.empty(10, 256))
    nn.init.trunc_normal_(checkpoint['state_dict']['fc_ks.weight'])

    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']

    print(f"Load model and optimizer state at iteration {iteration} of {filepath}")

def load_checkpoint_generator(model, filepath, iteration, device):
    checkpoint = torch.load(f'{filepath}/checkpoint_{iteration}', map_location=f'cuda:{device.index}')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_checkpoint_downstream(model, optimizer, filepath, device):
    #import pdb
    #pdb.set_trace()
    checkpoint = torch.load(filepath, map_location=f'cuda:{device.index}')
    del checkpoint['state_dict']['fc_sv.weight']
    del checkpoint['state_dict']['fc_sv.bias']
    #del checkpoint['state_dict']['fc_ks.weight']
    #del checkpoint['state_dict']['fc_ks.bias']
    #checkpoint['state_dict']['fc_ks.weight'] = torch.FloatTensor(torch.empty(10, 768))
    #checkpoint['state_dict']['fc_sv.bias'] = torch.FloatTensor(torch.empty(10))
    model.load_state_dict(checkpoint['state_dict'])

    print(f"Load model and optimizer state at {filepath}")

def get_mask_from_lengths(lengths):
    #import pdb
    #pdb.set_trace()
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len)).cuda()
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask


def reorder_batch(x, n_gpus):
    assert (x.size(0)%n_gpus)==0, 'Batch size must be a multiple of the number of GPUs.'
    new_x = x.new_zeros(x.size())
    chunk_size = x.size(0)//n_gpus
    
    for i in range(n_gpus):
        new_x[i::n_gpus] = x[i*chunk_size:(i+1)*chunk_size]
    
    return new_x

def copy_file(from_file_path, to_file_path):
    shutil.copyfile(from_file_path, to_file_path)

def data_preparation(process, config, data_name):
    if process == 'train':
        trainset = AudioSet('train', config)
        if data_name == 'libri':
            collate_fn   = AudioSetCollate()
            train_loader = DataLoader(trainset,
                                    shuffle=True,
                                    batch_size=config['train']['batch_size'], 
                                    collate_fn= collate_fn,
                                    drop_last=True)
        elif data_name == 'vox1' or data_name == 'KeywordSpotting':
            train_loader = DataLoader(trainset,
                                    shuffle=True,
                                    batch_size=config['train']['batch_size'], 
                                    drop_last=True)
        return train_loader
    elif process == 'val':
        valset = AudioSet('val', config)
        if data_name == 'libri':
            collate_fn = AudioSetCollate()
            val_loader = DataLoader(valset,
                                    shuffle=True,
                                    batch_size=1, 
                                    collate_fn=collate_fn,
                                    drop_last=True)
        elif data_name == 'vox1' or data_name == 'KeywordSpotting':
            val_loader = DataLoader(valset,
                                    shuffle=True,
                                    batch_size=1,
                                    drop_last=True)
        return val_loader
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import yaml

def accuracy(ind_estimated, ind_target):
    count   = torch.sum(ind_estimated == ind_target)
    results = count / (ind_estimated.shape[0] * ind_estimated.shape[1]) * 100
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
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']

    print(f"Load model and optimizer state at iteration {iteration} of {filepath}")

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

import torch

def add_gauss(x):
    return x + torch.normal(mean=0., std=1., size=x.size()).to(x.device)

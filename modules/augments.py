import torch

def add_gauss(x):
    x_pow = (x**2).mean()
    ns = torch.normal(mean=0., std=0.01, size=x.size()).to(x.device)
    corr_x = x + ns

    corr_x_pow = (corr_x**2).mean()
    scale = torch.sqrt(x_pow / corr_x_pow)
    return corr_x * scale

import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import Model
import yaml
from utils.data_utils import *
from utils.writer import get_writer
from utils.utils import *
import tqdm

def validate(model, criterion, val_loader, iteration, writer, device):

    model.eval()
    with torch.no_grad():

        n_data, val_mlm_loss, val_disc_loss = 0, 0, 0
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            if i == 10:
                break

            n_data += len(batch[0])
            wav_padded, wav_lengths, txt_padded, txt_lengths = [
                x.to(device) for x in batch
            ]
            mlm_loss, disc_loss, mlm_acc = model(wav_padded, wav_lengths, txt_padded, txt_lengths, criterion)
            
            val_mlm_loss  += mlm_loss.item() * len(batch[0])
            val_disc_loss += disc_loss.item() * len(batch[0])

        val_mlm_loss /= n_data
        val_disc_loss /= n_data

        print(f'|-Validation-| Iteration:{iteration} mlm loss:{val_mlm_loss:.3f} disc loss:{val_disc_loss:.3f} mlm_acc(%):{mlm_acc.item():.3f}')

    writer.add_losses(val_mlm_loss, iteration, 'Validation', 'mlm_loss')
    writer.add_losses(val_disc_loss, iteration, 'Validation', 'disc_loss')
    model.train()
    
    
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
    accumulation     = config['optimization']['accumulation']
    output_directory = config['train']['output_directory']
    output_name      = config['train']['output_name']

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
    if args.iteration != None:
        load_checkpoint(model, optimizer, args.iteration, f'{output_directory}/{output_name}')
        iteration += args.iteration

    model.train()
    print("|-Train-| Training Start!!!")
    while iteration < (train_steps * accumulation):
        for i, batch in enumerate(train_loader):
            wav_padded, wav_lengths, txt_padded, txt_lengths = [
                x.to(device) for x in batch
            ]

            mlm_loss, disc_loss, mlm_acc = model(wav_padded, wav_lengths, txt_padded, txt_lengths, criterion)
            tot_loss = mlm_loss + disc_loss

            sub_loss = (tot_loss)/accumulation
            sub_loss.backward()
            loss = loss+sub_loss.item()

            iteration += 1

            if iteration%accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                optimizer.step()
                optimizer.zero_grad()
                    
                writer.add_losses(mlm_loss.item(), iteration, 'Train', 'mlm_loss')
                writer.add_losses(disc_loss.item(), iteration, 'Train', 'disc_loss')
                print(f'|-Train-| Iteration:{iteration} mlm loss:{mlm_loss.item():.3f} disc loss:{disc_loss.item():.3f} mlm_acc(%):{mlm_acc.item():.3f}')
                loss=0

            if iteration%(iters_per_validation*accumulation)==0:
                validate(model, criterion, val_loader, iteration, writer, device)
                
                save_checkpoint(model,
                                optimizer,
                                lr,
                                iteration,
                                filepath=f'{output_directory}/{output_name}') # save file
                
            if iteration==(train_steps*accumulation):
                break

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
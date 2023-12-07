import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import Model
import yaml
from utils.data_utils import *
from utils.writer import get_writer
#from utils.build_and_load import *
import tqdm

def validate(model, generator, criterion, val_loader, iteration, writer):
    model.eval()
    generator.eval()
    with torch.no_grad():

        n_data, val_loss = 0, 0
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            n_data += len(batch[0])
            text_padded, text_lengths, mel_padded, mel_lengths, wav_padded, wav_lengths, list_spki = [
                x.to(model.device) for x in batch
            ]

            x_hat, y_hat, tokens, _ = generator(wav_padded, mode='eval')
            B, T, n_q  = tokens.size()
            
            lookuped_cw = generator.lookup_RVQ(tokens)  # [B, T', N_q] -> [B, T', N_q, D]
            y_hat_from_tokens = generator.codewords_sum_reduction(lookuped_cw)
            #y_hat_from_tokens = lookuped_cw[:,:,0].transpose(1,2)
            _, input_spec_dict = generator.prepare_input(wav_padded)
            x_hat  = generator.decoder(y_hat_from_tokens)
            x_hat  = generator.prepare_output(x_hat, input_spec_dict)

            y_estimated = model.outputs(text_padded,
                                        mel_padded,
                                        text_lengths,
                                        mel_lengths,
                                        list_spki)

            acc =accuracy(y_estimated, tokens[:,:,0])
            mel_loss    = criterion(y_estimated, tokens[:,:,0])
            y_estimated = torch.argmax(y_estimated,1)
            y_estimated = torch.repeat_interleave(y_estimated.unsqueeze(2), 4, dim=2)
            #_, all_losses = generator.quantizer(y_estimated.transpose(1,2), tokens)

            #y_hat, tokens, commit_loss = generator.quantizer(y_estimated.transpose(1,2))
            #_, tokens_hat   = torch.max(ctc_out, 1)  
            #tokens_hat      = torch.repeat_interleave(tokens_hat.unsqueeze(2), 8, dim=2)
            #tokens_hat     = torch.reshape(tokens_hat, (B, T, n_q))     
            lookuped_cw_hat = generator.lookup_RVQ(y_estimated)  # [B, T', N_q] -> [B, T', N_q, D]
            y_hat_from_tokens_hat = lookuped_cw_hat[:,:,0]
            
            #y_hat_from_tokens_hat = generator.codewords_sum_reduction(lookuped_cw_hat)   

            x_hat_hat = generator.decoder(y_hat_from_tokens_hat.transpose(1,2))
            x_hat_hat = generator.prepare_output(x_hat_hat, input_spec_dict)

            #mel_loss = criterion(y_hat.transpose(1,2), y_estimated, mel_lengths//2)
            #mel_loss = torch.mean(mel_loss)
            
            val_loss += mel_loss.item() * len(batch[0])

        val_loss /= n_data

        print(f'|-Validation-| Iteration:{iteration} ctc loss:{mel_loss.item():.3f} ACC(%):{acc:.3f}')

    writer.add_losses(mel_loss.item(), iteration, 'Validation', 'mel_loss')
    model.train()
    
    
def main(args):

    config_path = args.c
    with open(config_path) as fp:
        config = yaml.full_load(fp)


    train_steps = config['optimization']['train_steps']
    accumulation = config['optimization']['accumulation']
    iters_per_validation = config['optimization']['iters_per_validation']
    iters_per_checkpoint = config['optimization']['iters_per_checkpoint']
    grad_clip_thresh = config['optimization']['grad_clip_thresh']

    device    = torch.device(f'cuda:{str(args.gpu)}')

    trainset = AudioSet('train', config)
    collate_fn   = AudioSetCollate()
    train_loader = DataLoader(trainset,
                            shuffle=True,
                            batch_size=config['train']['batch_size'], 
                            collate_fn= collate_fn,
                            drop_last=True)

    valset = AudioSet('val', config)
    collate_fn   = AudioSetCollate()
    val_loader = DataLoader(valset,
                            shuffle=True,
                            batch_size=1, 
                            collate_fn=collate_fn,
                            drop_last=True)

    model     = Model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=384**-0.5,
                                 weight_decay=0.9)

    criterion = nn.CTCLoss(blank=0)

    writer = get_writer(config['train']['output_directory'], config['train']['output_name'])

    loss = 0
    iteration = 0
    model.train()
    print("|-Train-| Training Start!!!")
    while iteration < (train_steps*accumulation):
        for i, batch in enumerate(train_loader):
            wav_padded, wav_lengths, txt_padded, txt_lengths = [
                x.to(device) for x in batch
            ]

            ctc_loss, outputs = model(wav_padded, wav_lengths, txt_padded, txt_lengths, criterion)

            sub_loss = (ctc_loss)/accumulation
            sub_loss.backward()
            loss = loss+sub_loss.item()

            iteration += 1
            if iteration%accumulation == 0:
                lr_scheduling(optimizer, iteration//accumulation)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                optimizer.step()
                model.zero_grad()
                writer.add_losses(mel_loss.item(), iteration, 'Train', 'mel_loss')
                print(f'|-Train-| Iteration:{iteration} ctc loss:{mel_loss:.3f} ACC(%):{acc:.3f}')
                #writer.add_losses(all_losses.item(), iteration//hparams.accumulation, 'Train', 'commit_loss')
                loss=0
            if iteration%(hparams.iters_per_validation*hparams.accumulation)==0:
                y_estimated = model.outputs(ema_padded,
                                            mel_padded,
                                            ema_lengths,
                                            mel_lengths,
                                            list_spki)
                
                y_estimated = torch.argmax(y_estimated,1)
                y_estimated = torch.repeat_interleave(y_estimated.unsqueeze(2), 4, dim=2)
                with torch.no_grad():
                    #y_hat, tokens, _ = generator.quantizer(y_estimated.transpose(1,2))

                    lookuped_cw_hat  = generator.lookup_RVQ(y_estimated)
                    #y_hat_from_tokens_hat = generator.codewords_sum_reduction(lookuped_cw_hat)  
                    y_hat_from_tokens_hat  = lookuped_cw_hat[:,:,0]
                    _, input_spec_dict = generator.prepare_input(wav_padded)
                    x_hat_hat = generator.decoder(y_hat_from_tokens_hat.transpose(1,2))
                    x_hat_hat = generator.prepare_output(x_hat_hat, input_spec_dict) 
                
                writer.add_1d(wav_padded.detach().cpu(),
                            x_hat.detach().cpu(),
                            wav_lengths.detach().cpu(),
                            iteration//hparams.accumulation, 'Train', 'Raw waveform')

                writer.add_1d(x_hat.detach().cpu(),
                            x_hat_hat.detach().cpu(),
                            wav_lengths.detach().cpu(),
                            iteration//hparams.accumulation, 'Train', 'Quantized')
                
            if iteration%(hparams.iters_per_validation*hparams.accumulation)==0:
                validate(model, generator, criterion, val_loader, iteration, writer)
                """
                save_checkpoint(model,
                                optimizer,
                                hparams.lr,
                                iteration//hparams.accumulation,
                                filepath=f'{hparams.output_directory}/{hparams.log_directory}')
                """
            if iteration==(hparams.train_steps*hparams.accumulation):
                break

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--c', type=str, default='configs/default.yaml')
    args = p.parse_args()
    
    #config_path = '/home/miseul/cousework/it_hgkang/selectra/configs/default.yaml'
    config_path = args.c
    with open(config_path) as fp:
        config = yaml.full_load(fp)

    os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
    torch.manual_seed(config['train']['seed'])
    torch.cuda.manual_seed(config['train']['seed'])
        
    main(args)
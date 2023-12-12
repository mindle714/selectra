import os
from torch.utils.tensorboard import SummaryWriter

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        writer = Writer(logging_path)
        #raise Exception('The experiment already exists')
    else:
        os.makedirs(logging_path)
        writer = Writer(logging_path)
            
    return writer


class Writer(SummaryWriter):
    def __init__(self, log_dir):
        super(Writer, self).__init__(log_dir)

    def add_losses(self, mel_loss, global_step, phase, name_loss):
        self.add_scalar(f'{phase}/{name_loss}', mel_loss, global_step)

    def add_sounds(self, target, output, iteration, mode, sample_rate): 
        B = target.size(0)
        if B > 10:
            N = 10
        else:
            N = B
        for i in range(N):  
            self.add_audio(f'{mode}/target_{i+1}', target[i].unsqueeze(0), iteration, sample_rate=sample_rate)
            self.add_audio(f'{mode}/output_{i+1}', output[i].unsqueeze(0), iteration, sample_rate=sample_rate)

    def add_specs(self, mel_padded, mel_out, mel_out_post, mel_lengths, global_step, phase):
        N = mel_padded.size(0)
        if N > 10:
            N = 10
        for i in range(N):
            mel_fig = plot_melspec(mel_padded, mel_out, mel_out_post, mel_lengths, i)    
            self.add_figure(f'melspec/{phase}_{i}', mel_fig, global_step)

    def add_1d(self, e_padded, e_out, mel_lengths, global_step, phase, name_loss):
        N = e_padded.size(0)
        if N > 10:
            N = 10
        for i in range(N):
            e_fig = plot_1d(e_padded, e_out, mel_lengths, i)    
            self.add_figure(f'{name_loss}/{phase}_{i}', e_fig, global_step)
        
    def add_alignments(self, enc_alignments, dec_alignments,
                       text_padded, mel_lengths, text_lengths, global_step, phase):
        N = enc_alignments.size(0)
        for i in range(N):
            enc_align_fig = plot_alignments(enc_alignments, text_padded, mel_lengths, text_lengths, 'enc', i)
            self.add_figure(f'enc_alignments/{phase}_{i}', enc_align_fig, global_step)

            dec_align_fig = plot_alignments(dec_alignments, text_padded, mel_lengths, text_lengths, 'dec', i)
            self.add_figure(f'dec_alignments/{phase}_{i}', dec_align_fig, global_step)

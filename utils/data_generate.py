import yaml
import glob 
import librosa 
import soundfile as sf
import os
from text import text_to_sequence
import random

config_path = '/home/miseul/cousework/it_hgkang/selectra/configs/default.yaml'
with open(config_path) as fp:
    config = yaml.full_load(fp)

random.seed(config['train']['seed'])
training_ratio   = 0.9
validation_ratio = 0.1
training_files   = config['train']['training_files']
validation_files = config['train']['validation_files']
data_path        = config['train']['data_path']

list_wav = glob.glob(f'{data_path}/*/*/*.flac')
list_wav = sorted(list_wav)
list_spk = glob.glob(f'{data_path}/*/*')
print(f'Number of dataset:{len(list_wav)}')
print(f'Number of dataset:{len(list_spk)}')

spk_dict = dict()
tot_txt  = ''
for path_spk in list_spk:
    tmp = path_spk.split('/')
    path_txt = os.path.join(path_spk, f'{tmp[-2]}-{tmp[-1]}.trans.txt')
    with open(path_txt, 'r') as f :
        lines = f.readlines()
        for line in lines:
            file_name = line.split()[0]
            script = ' '.join(line.split()[1:])
            tot_txt += script
            script_id = text_to_sequence(script, ['custom_english_cleaners'])
            spk_dict[file_name] = script

tot_txt = set(tot_txt)
tot_txt = sorted(tot_txt)[2:]
print(tot_txt)
print(len(tot_txt))
N = len(tot_txt) + 1

random.shuffle(list_wav)
list_train = list_wav[:int(len(list_wav) * training_ratio)]
list_val   = list_wav[int(len(list_wav) * training_ratio):]
print(f' Number of training files: {len(list_train)}')
print(f' Number of validation files: {len(list_val)}')

def generate_filetxt(filename, list):
    with open(filename, 'w') as f:
        for path_wav in list:
            script_name = path_wav.split('/')[-1].replace('.flac', '')
            script = spk_dict[script_name]
            #print(script)
            print(path_wav)
            path_wav_write = '/'.join(path_wav.split('/')[-3:])

            f.write(f'{path_wav_write}|{script}\n')
            #f.write()

generate_filetxt(training_files, list_train)
generate_filetxt(validation_files, list_val)
"""
out = text_to_sequence('MISEUL', ['custom_english_cleaners'])
print(len('MISEUL'))
print(len(out))
"""
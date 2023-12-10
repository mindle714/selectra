# SELECTRA

SELECTRA : Speech ELECTRA

## How to train

```
pip3 install -r requirements.txt
python3 train.py --gpu {gpu index} --c {path of config file} --iteration {load iteration of pre-trained model}
```

## How to check tensorboard

```
python3 -m tensorboard.main --logdir {directory} --port {number of port} --bind_all
or
tensorboard --logdir {directory} --port {number of port} --bind_all
```

from datasets import load_dataset
from models import load_model

from utils import train_one_epoch, save_model_ckpt, save_loss_ckpt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse, time, os, sys, yaml

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)

    return parser
        
def main(cfg):
    print(f"=====================[{cfg['title']}]=====================")

    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else: 
        device = 'cpu'
    print(f"device: {device}")

    # Hyperparameter Settings
    hp_cfg = cfg['hyperparameters']

    # Load Dataset
    train_data_cfg = cfg['data']['train']
    train_ds = load_dataset(train_data_cfg)
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           shuffle=True,
                                           batch_size=hp_cfg['batch_size'],
                                           drop_last=True)
    print(f"Load Train Dataset {train_data_cfg['name']}")
            
    # Load Model
    model_cfg = cfg['model']
    print(model_cfg['name'])
    model = load_model(model_cfg).to(device)

    print(model_cfg['img_dim'])
    
    if cfg['parallel'] == True:
        model = nn.DataParallel(model)
    
    # Loss Function
    if hp_cfg['loss_fn'] == 'BCE':
        # 이미 Discriminator의 마지막에 Sigmoid가 있으므로, BCEWithLogitsLoss 사용 X
        loss_fn = nn.BCELoss()
    else:
        raise Exception(f"Check loss function in configuration file")
    
    # Optimizer
    optimizer = None
    if hp_cfg['optim'] == "SGD" and model_cfg['name'] in ['GAN']:
        optimizer = {
            'gen_optim': optim.SGD(model.generator.parameters(), lr=hp_cfg['gen_lr'], momentum=hp_cfg['momentum']),
            'disc_optim': optim.SGD(model.discriminator.parameters(), lr=hp_cfg['disc_lr'], momentum=hp_cfg['momentum'])
        }

    # Load Scheduler
    if model_cfg['name'] in ['GAN']:
        scheduler = {
            'gen_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer['gen_optim'],
                                                                   mode='min',
                                                                   factor=0.5,
                                                                   patience=5,
                                                                   min_lr=1e-6),
            'disc_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer['disc_optim'],
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=5,
                                                                    min_lr=1e-6)
        }
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.5,
                                                        patience=5,
                                                        min_lr=1e-6)
    
    task_cfg = cfg['task']
    save_cfg = cfg['save']

    # Training loss
    total_train_loss = []
    total_start_time = int(time.time())
    
    min_loss = 1e4
    
    for current_epoch in range(1, hp_cfg['epochs']+1):
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{hp_cfg['epochs']:03d}]\n")
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, scheduler, task_cfg, device)
        elapsed_time = int(time.time() - start_time)
        print(f"Train Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s\n")

        # Validation을 따로 하지는 않을 것. 추후에 FID, IS를 고려할 수 있으나, 지금은 생략.
        # val_loss = validate(model, val_dl, loss_fn, scheduler, task_cfg, device) # input args

        if train_loss[0] < min_loss and current_epoch > 50:
            min_loss = train_loss[0]
            save_model_ckpt(model, save_cfg['name'], current_epoch, save_cfg['weights_path'])

        total_train_loss.append(train_loss)
        save_loss_ckpt(save_cfg['name'], total_train_loss, save_cfg['loss_path'])

    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/train/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)
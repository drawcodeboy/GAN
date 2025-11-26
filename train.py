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
    
    # Optimizer
    optimizer = None
    if hp_cfg['optim'] == "SGD" and model_cfg['name'] in ['GAN']:
        optimizer = {
            'd_optim': optim.SGD(model.discriminator.parameters(), lr=hp_cfg['disc_lr'], momentum=hp_cfg['momentum']),
            'g_optim': optim.SGD(model.generator.parameters(), lr=hp_cfg['gen_lr'], momentum=hp_cfg['momentum'])
        }
    
    task_cfg = cfg['task']
    save_cfg = cfg['save']

    # Training loss
    total_g_loss = []
    total_d_loss = []
    total_start_time = int(time.time())
    
    for current_epoch in range(1, hp_cfg['epochs']+1):
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{hp_cfg['epochs']:03d}]\n")
        
        # Training One Epoch
        start_time = int(time.time())
        d_loss, g_loss = train_one_epoch(model, train_dl, optimizer, task_cfg, device)
        elapsed_time = int(time.time() - start_time)
        print(f"Train Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s\n")
        
        if current_epoch % 50 == 0:
            save_model_ckpt(model, save_cfg['name'], current_epoch, save_cfg['weights_path'])
            
        total_g_loss.append(g_loss)
        total_d_loss.append(d_loss)
        save_loss_ckpt(save_cfg['name'], "generator", total_g_loss, save_cfg['loss_path'])
        save_loss_ckpt(save_cfg['name'], "discriminator", total_d_loss, save_cfg['loss_path'])
    save_model_ckpt(model, save_cfg['name'], current_epoch, save_cfg['weights_path'])

    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/train/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)
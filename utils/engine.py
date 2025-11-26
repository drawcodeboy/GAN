import torch
import torch.nn.functional as F

import numpy as np
import pdb
import random

from .loss_process import adversarial_process

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, task_cfg, device):
    model.train()
    total_loss = []
    total_g_loss, total_d_loss = [], []
    
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'adversarial_learning':
            g_loss, d_loss = adversarial_process(model, data, loss_fn, optimizer, device)
            total_g_loss.append(g_loss)
            total_d_loss.append(d_loss)

        else:
            raise Exception("Check your task_cfg['object'] configuration")
         
        if task_cfg['object'] == 'adversarial_learning':
            print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, "
                  f"Generator Loss: {sum(total_g_loss)/len(total_g_loss):.6f}, "
                  f"Discriminator Loss: {sum(total_d_loss)/len(total_d_loss):.6f}, "
                  f"Generator LR: {scheduler['gen_scheduler'].get_last_lr()[0]:.6f} "
                  f"Discriminator LR: {scheduler['disc_scheduler'].get_last_lr()[0]:.6f}", end="")
    print()
    
    if task_cfg['object'] == 'adversarial_learning':
        return [sum(total_g_loss)/len(total_g_loss), sum(total_d_loss)/len(total_d_loss)]

'''
@torch.no_grad()
def evaluate(model, dataloader, task_cfg, device):
    model.eval()
    
    total_x, total_x_prime, total_codes, total_factors = [], [], [], []
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'test_vae':
            x, label = data
            x = x.to(device)           
            label = label.to(device)

            x_prime, z, mu, log_var = model(x)
            
            total_x.append(x.cpu().numpy())
            total_x_prime.append(x_prime.cpu().numpy())
            total_codes.append(mu.cpu().numpy()) # z 대신 mu
            total_factors.append(label.cpu().numpy())
            
        else:
            raise Exception("Check your task_cfg['object'] configuration")
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    total_x = np.concatenate(total_x)
    total_x_prime = np.concatenate(total_x_prime)
    total_codes = np.concatenate(total_codes)
    total_factors = np.concatenate(total_factors)
    
    result = get_metrics(total_x, total_x_prime, total_codes, total_factors)
    
    return result
'''
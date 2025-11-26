import torch
import numpy as np
import os

def save_model_ckpt(model, model_name, current_epoch, save_dir):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    
    save_name = f"{model_name}.epochs_{current_epoch:03d}.pth"
    
    try:
        torch.save(ckpt, os.path.join(save_dir, save_name))
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")
        
def save_loss_ckpt(model_name, model_type, train_loss, save_dir):
    try:
        np.save(os.path.join(save_dir, f'train_loss_{model_type}_{model_name}.npy'), np.array(train_loss))
        print(f'Save {model_type} Train Loss')
    except:
        print('Can\'t Save Train Loss') 
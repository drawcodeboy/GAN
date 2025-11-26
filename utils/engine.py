def train_one_epoch(model, dataloader, optimizer, task_cfg, device):
    model.train()
    total_g_loss, total_d_loss = [], []
    
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'adversarial_learning':
            x = data.to(device)
            g_loss, d_loss = model.adv_loss_update(x, optimizer['d_optim'], optimizer['g_optim'])
            total_g_loss.append(g_loss)
            total_d_loss.append(d_loss)

        else:
            raise Exception("Check your task_cfg['object'] configuration")
         
        if task_cfg['object'] == 'adversarial_learning':
            print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, "
                  f"Discriminator Loss: {sum(total_d_loss)/len(total_d_loss):.6f}, "
                  f"Generator Loss: {sum(total_g_loss)/len(total_g_loss):.6f}", end="")
    print()
    
    if task_cfg['object'] == 'adversarial_learning':
        return [sum(total_d_loss)/len(total_d_loss), sum(total_g_loss)/len(total_g_loss)]
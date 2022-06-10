import os
import yaml
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import get_data
from model import BERT_VAE


# closed form kl loss computation between variational posterior q(z|x) and unit Gaussian prior p(z) 
def kl_loss_fn(z_mu,z_rho):
    sigma_squared = F.softplus(z_rho) ** 2
    kl_1d = -0.5 * (1 + torch.log(sigma_squared) - z_mu ** 2 - sigma_squared)

    # sum over sample dim, average over batch dim
    kl_batch = torch.mean(torch.sum(kl_1d,dim=1))

    return kl_batch


def train(cfg, device, performance_logger):

    train_dataloader, tokenizer = get_data(cfg, split=0)
    performance_logger.info('train data loaded.')

    valid_dataloader, _ = get_data(cfg, split=1)
    performance_logger.info('valid data loaded.\n')

    net = BERT_VAE(cfg).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scaler = GradScaler()
    
    lowest_train_loss = float('inf') # should be val loss for checkpointing

    # zero the parameters' gradients
    optimizer.zero_grad()

    for epoch in range(cfg['epochs']):  # loop over dataset

        net.train()
        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')
        
        batch_perplexity_array=[]
        batch_rec_loss_array=[]
        batch_kl_loss_array=[]


        total_iterations = int(len(train_dataloader) / cfg['gradient_accumulations'])

        # training
        for batch_idx, batch_data in enumerate(train_dataloader): # loop over train batches
            
            batch_data = batch_data.to(device)
            input_ids = batch_data['input_ids']
            attention_mask = batch_data['attention_mask']

            # forward pass
            with autocast():
                z_mu, z_rho, preds = net(batch_data)
                preds = preds.permute((0,2,1))

                # modify prediction and ground truth tensor for autoregressive loss
                # do not include padding for loss computation
                y_true = input_ids[:,1:]
                y_pred = preds[:,:,:-1]
                binary_loss_mask = attention_mask[:,1:]==1.0

                loss_tensor = loss_fn(input=y_pred, target=y_true)
                masked_loss_tensor = loss_tensor.where(binary_loss_mask, torch.tensor(0.0).to(device))
                reconstruction_loss = masked_loss_tensor.sum() / binary_loss_mask.sum()

                kl_loss = cfg['kl_weight'] * kl_loss_fn(z_mu, z_rho)

                elbo = reconstruction_loss + kl_loss

            # backpropagation
            scaler.scale(elbo / cfg['gradient_accumulations']).backward()

            # gradient descent with optimizer
            if (batch_idx + 1) % cfg['gradient_accumulations'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # save batch metrics
            detached_rec_loss = reconstruction_loss.detach().cpu()
            batch_rec_loss_array.append(detached_rec_loss.item())
            batch_kl_loss_array.append(kl_loss.detach().cpu().item())
            batch_perplexity_array.append(torch.exp(detached_rec_loss).item())

            # print intermediate iterations during epoch
            if ((batch_idx + 1) / cfg['gradient_accumulations']) % cfg['print_iter_freq'] == 0:
                intermediate_batch_rec_loss = np.mean(batch_rec_loss_array[-cfg['gradient_accumulations']:])
                intermediate_batch_kl_loss = np.mean(batch_kl_loss_array[-cfg['gradient_accumulations']:])
                intermediate_batch_perplexity = np.mean(batch_perplexity_array[-cfg['gradient_accumulations']:])
                performance_logger.info(f'iter: {int((batch_idx + 1) / cfg["gradient_accumulations"])} / {total_iterations}, iter_rec_loss: {intermediate_batch_rec_loss:.4f}, iter_kl_loss: {intermediate_batch_kl_loss:.4f}, iter_perplexity: {intermediate_batch_perplexity:.4f}')               


        # validation
        '''net.eval()
        with torch.no_grad():

            batch_perplexity_array_valid=[]
            batch_loss_array_valid=[]

            for _, valid_batch_data in enumerate(valid_dataloader): # loop over valid batches
                
                valid_batch_data = valid_batch_data.to(device)

                # forward pass with mixed precision
                with autocast():
                    val_loss,_ = net(valid_batch_data)

                # save batch metrics
                detached_val_loss = val_loss.detach().cpu()
                batch_loss_array_valid.append(detached_val_loss.item())
                batch_perplexity_array_valid.append(torch.exp(detached_val_loss).item())'''


        # display metrics at end of epoch
        epoch_train_rec_loss, epoch_train_kl_loss, epoch_train_perplexity = np.mean(batch_rec_loss_array), np.mean(batch_kl_loss_array), np.mean(batch_perplexity_array)
        #epoch_val_loss, epoch_val_perplexity = np.mean(batch_loss_array_valid), np.mean(batch_perplexity_array_valid)

        #performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}, val_loss: {epoch_val_loss:.4f}, val_perplexity: {epoch_val_perplexity:.4f}\n')
        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_rec_loss: {epoch_train_rec_loss:.4f}, train_kl_loss: {epoch_train_kl_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}\n')



        # save if best
        #if lowest_train_loss > epoch_train_loss:
        #    lowest_train_loss = epoch_train_loss
        # save only for selected epochs
        #if epoch > 0 and epoch%1==0:
        save_dict = {'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_train_rec_loss + epoch_train_kl_loss}
        torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_epoch'+str(epoch)+'.pt'))
    
    
    return




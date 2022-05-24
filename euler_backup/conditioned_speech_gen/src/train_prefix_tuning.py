import os
import yaml
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer

from dataset import get_data
from model import PrefixNet


def train(cfg, device, performance_logger):

    train_dataloader, tokenizer = get_data(cfg, split=0)
    performance_logger.info('train data loaded.')

    valid_dataloader, _ = get_data(cfg, split=1)
    performance_logger.info('valid data loaded.\n')

    net = PrefixNet(cfg).to(device)

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
        batch_loss_array=[]

        total_iterations = int(len(train_dataloader) / cfg['gradient_accumulations'])

        # training
        for batch_idx, batch_data in enumerate(train_dataloader): # loop over train batches
            
            batch_data = batch_data.to(device)
            encoded_bioguide_ids = batch_data[cfg['encoded']]
            input_ids = batch_data['input_ids']
            attention_mask = batch_data['attention_mask']

            # 1st forward pass
            with torch.no_grad():
                # one forward pass for obtaining token embeddings
                o = net.gpt_neo(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # get embedding layer's output which is wte + wpe
            wte_plus_wpe = o.hidden_states[0]
            # remove positional embeddings
            just_wte = wte_plus_wpe - torch.unsqueeze(net.gpt_neo.transformer.wpe(torch.arange(start=0,end=wte_plus_wpe.shape[1]).to(device)),0)

            # 2nd forward pass with mixed precision
            with autocast():
                preds = net(just_wte, encoded_bioguide_ids)
                preds = preds.permute((0,2,1))

                # modify prediction and ground truth tensor for autoregressive loss
                # do not include padding and prefix tokens for loss computation
                # although prefix tokens are learnable
                y_true = input_ids[:,1:]
                y_pred = preds[:,:,cfg['prefix_len']:-1]
                binary_loss_mask = attention_mask[:,1:]==1.0

                loss_tensor = loss_fn(input=y_pred, target=y_true)
                masked_loss_tensor = loss_tensor.where(binary_loss_mask, torch.tensor(0.0).to(device))
                final_loss = masked_loss_tensor.sum() / binary_loss_mask.sum()

            # backpropagation
            scaler.scale(final_loss / cfg['gradient_accumulations']).backward()

            # gradient descent with optimizer
            if (batch_idx + 1) % cfg['gradient_accumulations'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # save batch metrics
            detached_loss = final_loss.detach().cpu()
            batch_loss_array.append(detached_loss.item())
            batch_perplexity_array.append(torch.exp(detached_loss).item())

            # print intermediate iterations during epoch
            if ((batch_idx + 1) / cfg['gradient_accumulations']) % cfg['print_iter_freq'] == 0:
                intermediate_batch_loss = np.mean(batch_loss_array[-cfg['gradient_accumulations']:])
                intermediate_batch_perplexity = np.mean(batch_perplexity_array[-cfg['gradient_accumulations']:])
                performance_logger.info(f'iter: {int((batch_idx + 1) / cfg["gradient_accumulations"])} / {total_iterations}, iter_loss: {intermediate_batch_loss:.4f}, iter_perplexity: {intermediate_batch_perplexity:.4f}')               


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
        epoch_train_loss, epoch_train_perplexity = np.mean(batch_loss_array), np.mean(batch_perplexity_array)
        #epoch_val_loss, epoch_val_perplexity = np.mean(batch_loss_array_valid), np.mean(batch_perplexity_array_valid)

        #performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}, val_loss: {epoch_val_loss:.4f}, val_perplexity: {epoch_val_perplexity:.4f}\n')
        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}\n')



        # save if best
        #if lowest_train_loss > epoch_train_loss:
        #    lowest_train_loss = epoch_train_loss
        # save only for selected epochs
        if epoch > 0 and epoch%3==0:
            save_dict = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss}
            torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_epoch'+str(epoch)+'.pt'))
    
    
    return




import os
import yaml
import logging
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2Tokenizer

from dataset import get_data
from model import Net


# this function is only for observing finetuning progress by generating some free text
# it has no usage in our pipeline
def generate_text(cfg, model, tokenizer, generated_text_logger):

    # beam search + sampling
    gen_tokens = model.gpt_neo.generate(
        num_beams=5, 
        do_sample = True,
        early_stopping=True,
        max_length=cfg['max_seq_len'],
        num_return_sequences=1
    )
    gen_text = tokenizer.decode(gen_tokens[0],skip_special_tokens=True)
    generated_text_logger.info(gen_text)

    # nucleus sampling
    gen_tokens = model.gpt_neo.generate(
        do_sample=True,
        max_length=cfg['max_seq_len'],
        top_k=100,
        top_p=0.9,
        num_return_sequences=1
    )
    gen_text = tokenizer.decode(gen_tokens[0],skip_special_tokens=True)
    generated_text_logger.info(gen_text)

    return


# main function responsible for fine-tuning
def train(cfg, device, performance_logger, generated_text_logger):

    train_dataloader, tokenizer = get_data(cfg, split=0)
    performance_logger.info('train data loaded.')

    valid_dataloader, _ = get_data(cfg, split=1)
    performance_logger.info('valid data loaded.\n')

    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scaler = GradScaler()

    # zero the parameters' gradients
    optimizer.zero_grad()

    # generate some text before finetuning starts
    generated_text_logger.info('before fine-tuning')
    generate_text(cfg, net, tokenizer, generated_text_logger)
    generated_text_logger.info('-----')


    total_iters = 0

    for epoch in range(cfg['epochs']):  # loop over dataset

        net.train()
        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')
        
        batch_perplexity_array=[]
        batch_loss_array=[]

        total_iterations = int(len(train_dataloader) / cfg['gradient_accumulations'])

        # training
        for batch_idx, batch_data in enumerate(train_dataloader): # loop over train batches
            
            batch_data = batch_data.to(device)

            # forward pass with mixed precision
            with autocast():
                loss,_ = net(batch_data)

            # backpropagation
            scaler.scale(loss / cfg['gradient_accumulations']).backward()

            # gradient descent with optimizer
            if (batch_idx + 1) % cfg['gradient_accumulations'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # save batch metrics
            detached_loss = loss.detach().cpu()
            batch_loss_array.append(detached_loss.item())
            batch_perplexity_array.append(torch.exp(detached_loss).item())

            # print intermediate iterations during epoch
            if ((batch_idx + 1) / cfg['gradient_accumulations']) % cfg['print_iter_freq'] == 0:
                total_iters += int(cfg['print_iter_freq'])

                intermediate_batch_loss = np.mean(batch_loss_array[-cfg['gradient_accumulations']:])
                intermediate_batch_perplexity = np.mean(batch_perplexity_array[-cfg['gradient_accumulations']:])
                performance_logger.info(f'iter: {int((batch_idx + 1) / cfg["gradient_accumulations"])} / {total_iterations}, iter_loss: {intermediate_batch_loss:.4f}, iter_perplexity: {intermediate_batch_perplexity:.4f}')               
                
                # generate some text between iterations
                generated_text_logger.info(f'iter: {int((batch_idx + 1) / cfg["gradient_accumulations"])} / {total_iterations}')
                generate_text(cfg, net, tokenizer, generated_text_logger)
                generated_text_logger.info('-----')


        # validation
        net.eval()
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
                batch_perplexity_array_valid.append(torch.exp(detached_val_loss).item())


            # generate some text between epochs
            generated_text_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')
            generate_text(cfg, net, tokenizer, generated_text_logger)
            generated_text_logger.info('-----')
            

        # display metrics at end of epoch
        epoch_train_loss, epoch_train_perplexity = np.mean(batch_loss_array), np.mean(batch_perplexity_array)
        epoch_val_loss, epoch_val_perplexity = np.mean(batch_loss_array_valid), np.mean(batch_perplexity_array_valid)

        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}, val_loss: {epoch_val_loss:.4f}, val_perplexity: {epoch_val_perplexity:.4f}\n')


        # save every epoch
        save_dict = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss}
        torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_epoch'+str(epoch)+'.pt'))
    
    
    return




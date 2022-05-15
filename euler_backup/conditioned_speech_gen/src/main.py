import os
import yaml
import logging
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer

from dataset import get_data
from model import Net

def train(cfg, device):

    train_dataloader, tokenizer = get_data(cfg, split=0)
    performance_logger.info('train data loaded.\n')

    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scaler = GradScaler()
    # train and validation writers will be different
    #train_writer = SummaryWriter(log_dir=os.path.join(cfg['log_dir'],cfg['experiment_name'],'train'))
    
    lowest_train_loss = float('inf')

    # zero the parameters' gradients
    optimizer.zero_grad()

    for epoch in range(cfg['epochs']):  # loop over dataset

        net.train()
        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')
        
        batch_perplexity_array=[]
        batch_loss_array=[]

        total_iterations = int(len(train_dataloader) / cfg['gradient_accumulations'])

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
                intermediate_batch_loss = np.mean(batch_loss_array[-cfg['gradient_accumulations']:])
                intermediate_batch_perplexity = np.mean(batch_perplexity_array[-cfg['gradient_accumulations']:])
                performance_logger.info(f'iter: {int((batch_idx + 1) / cfg["gradient_accumulations"])} / {total_iterations}, iter_loss: {intermediate_batch_loss:.4f}, iter_perplexity: {intermediate_batch_perplexity:.4f}')               
                #train_writer.add_scalar('iter_loss', intermediate_batch_loss, int((batch_idx + 1) / cfg["gradient_accumulations"]) + epoch*total_iterations)
                #train_writer.add_scalar('iter_perplexity', intermediate_batch_perplexity, int((batch_idx + 1) / cfg["gradient_accumulations"]) + epoch*total_iterations)

        # text_generation
        net.eval()
        with torch.no_grad():
            #speaker_name_prompts = ['Neil Abercrombie ', 'Travis Childers ', 'Harry Reid ', 'Mitch McConnell ', 'Joseph Heck ']
            speaker_name_prompts = ['Joseph Heck', 'Travis Childers']

            generated_text_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}')

            for i in range(len(speaker_name_prompts)):
                inputs = tokenizer(speaker_name_prompts[i], return_tensors="pt").to(device)

                gen_tokens = net.gpt_neo.generate(
                    inputs.input_ids,
                    do_sample=True,
                    temperature=0.9,
                    max_length=cfg['max_seq_len'],
                    repetition_penalty=1.0,
                )
                gen_text = tokenizer.decode(gen_tokens[0],skip_special_tokens=True)
                
                generated_text_logger.info(gen_text)
            
            generated_text_logger.info('-----')
            

        # display metrics at end of epoch
        epoch_train_loss, epoch_train_perplexity = np.mean(batch_loss_array), np.mean(batch_perplexity_array)
        performance_logger.info(f'epoch: {epoch+1} / {cfg["epochs"]}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}\n')
        #train_writer.add_scalar('epoch_loss', epoch_train_loss, epoch)
        #train_writer.add_scalar('epoch_perplexity', epoch_train_perplexity, epoch)


        # save if best
        '''
        if lowest_train_loss > epoch_train_loss:
            lowest_train_loss = epoch_train_loss
            save_dict = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss}
            torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_best.pt'))'''
    
    
    return



if __name__ == '__main__':

    # load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    with open(_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # create logger
    #logging.basicConfig(filename=os.path.join(cfg['log_dir'],cfg['experiment_name']), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d-%m-%Y %H:%M')
    
    #performance_logger = logging.getLogger(os.path.join(cfg['log_dir'],cfg['experiment_name']))
    #generated_text_logger = logging.getLogger(os.path.join(cfg['log_dir'],cfg['experiment_name']+'2'))
    
    performance_logger = logging.getLogger('perf_logger')
    performance_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  - %(message)s','%d-%m-%Y %H:%M')
    perf_file_handler = logging.FileHandler(os.path.join(cfg['log_dir'],cfg['experiment_name']+'_performance'))
    perf_file_handler.setLevel(logging.INFO)
    perf_file_handler.setFormatter(formatter)
    performance_logger.addHandler(perf_file_handler)

    generated_text_logger = logging.getLogger('gen_logger')
    generated_text_logger.setLevel(logging.INFO)
    formatter2 = logging.Formatter('%(message)s')
    gen_file_handler = logging.FileHandler(os.path.join(cfg['log_dir'],cfg['experiment_name']+'_generated_texts'))
    gen_file_handler.setLevel(logging.INFO)
    gen_file_handler.setFormatter(formatter2)
    generated_text_logger.addHandler(gen_file_handler)


    performance_logger.info(f'cfg: {cfg}')
    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    performance_logger.info(f'Using device: {device}')

    train(cfg, device)
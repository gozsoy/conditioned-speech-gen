import os
import yaml
import logging
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from dataset import get_data
from model import Net

def train(cfg, device):

    train_dataloader = get_data(cfg, split=0)
    logging.info('train data loaded.')

    # TODO: add callbacks (tensorboard, lr scheduler, early stopper, checkpoint)
    # TODO: generate text per epoch -> tensorboard

    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scaler = GradScaler()

    lowest_train_loss = float('inf')

    # zero the parameters' gradients
    optimizer.zero_grad()

    for epoch in range(cfg['epochs']):  # loop over dataset

        batch_perplexity_array=[]
        batch_loss_array=[]

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

        # display metrics at end of epoch
        epoch_train_loss, epoch_train_perplexity = np.mean(batch_loss_array), np.mean(batch_perplexity_array)
        logging.info(f'epoch: {epoch}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}')

        # save last and best -if lowest so far-
        '''save_dict = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss}
        torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_last.pt'))
        
        if lowest_train_loss > epoch_train_loss:
            lowest_train_loss = epoch_train_loss
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
    logging.basicConfig(filename=os.path.join(cfg['log_dir'],cfg['experiment_name']), level=logging.INFO)
    logging.info(f'cfg: {cfg}')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')


    train(cfg, device)
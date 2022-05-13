import os
import argparse
import torch
import torch.optim as optim
import logging
import yaml
import numpy as np

from dataset import get_data
from model import Net

def train(cfg, device):

    train_dataloader = get_data(cfg, split=0)
    logging.info('train data loaded.')

    # TODO: add callbacks (tensorboard, lr scheduler, early stopper, checkpoint)
    # TODO: generate text per epoch -> tensorboard

    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])

    lowest_train_loss = float('inf')

    for epoch in range(cfg['epochs']):  # loop over dataset

        batch_perplexity_array=[]
        batch_loss_array=[]

        for batch_data in train_dataloader: # loop over train batches

            # zero the parameters' gradients
            optimizer.zero_grad()

            # forward pass
            loss,logits = net(batch_data.to(device))

            # backpropagation
            loss.backward()

            # gradient descent with optimizer
            # TODO: gradient clipping here
            optimizer.step()

            # save batch metrics
            detached_loss = loss.detach().cpu()
            batch_loss_array.append(detached_loss.item())
            # TODO: probably wrong computation. check https://huggingface.co/docs/transformers/perplexity
            batch_perplexity_array.append(torch.exp(detached_loss).item())
        
        # display metrics at end of epoch
        epoch_train_loss, epoch_train_perplexity = np.mean(batch_loss_array), np.mean(batch_perplexity_array)
        logging.info(f'epoch: {epoch}, train_loss: {epoch_train_loss:.4f}, train_perplexity: {epoch_train_perplexity:.4f}')

        # save last and best -if lowest so far-
        save_dict = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss}
        torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_last.pt'))
        
        if lowest_train_loss > epoch_train_loss:
            lowest_train_loss = epoch_train_loss
            torch.save(save_dict, os.path.join(cfg['checkpoint_dir'],cfg['experiment_name']+'_best.pt'))
    
    
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
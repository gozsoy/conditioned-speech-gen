import os
import yaml
import logging
import argparse
import numpy as np

import torch

import train_prefix_tuning
import train_speaker_prompt
import train_bert_vae

if __name__ == '__main__':

    # load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    with open(_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # create loggers
    performance_logger = logging.getLogger(cfg['experiment_name']+cfg['model']+'_perf_logger')
    performance_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  - %(message)s','%d-%m-%Y %H:%M')
    perf_file_handler = logging.FileHandler(os.path.join(cfg['log_dir'],cfg['model'],cfg['experiment_name']+'_performance'))
    perf_file_handler.setLevel(logging.INFO)
    perf_file_handler.setFormatter(formatter)
    performance_logger.addHandler(perf_file_handler)

    generated_text_logger = logging.getLogger(cfg['experiment_name']+cfg['model']+'_gen_logger')
    generated_text_logger.setLevel(logging.INFO)
    formatter2 = logging.Formatter('%(message)s')
    gen_file_handler = logging.FileHandler(os.path.join(cfg['log_dir'],cfg['model'],cfg['experiment_name']+'_generated_texts'))
    gen_file_handler.setLevel(logging.INFO)
    gen_file_handler.setFormatter(formatter2)
    generated_text_logger.addHandler(gen_file_handler)

    # print settings
    performance_logger.info(f'cfg: {cfg}')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    performance_logger.info(f'Using device: {device}')

    if cfg['model']=='speaker_prompt' or cfg['model']=='k2t':
        train_speaker_prompt.train(cfg, device, performance_logger, generated_text_logger)
    elif cfg['model']=='prefix_tuning':
        train_prefix_tuning.train(cfg,device,performance_logger)
    elif cfg['model']=='bert_vae':
        train_bert_vae.train(cfg,device,performance_logger)
    else:
        raise NotImplementedError()
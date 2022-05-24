import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import GPT2Tokenizer


def get_data(cfg, split=0):

    # read whole corpus
    if split == 0:
        processed_df = pd.read_csv(os.path.join(cfg['data_path'],'processed_df_train.csv'))
    elif split == 1:
        processed_df = pd.read_csv(os.path.join(cfg['data_path'],'processed_df_valid.csv'))
    else:
        raise Exception('wrong dataset split')        

    # TEMPORARY: work on subset of speakers
    #processed_df = processed_df[processed_df['encoded_bioguide_ids'].isin(list(np.arange(0,cfg['speaker_size'])))]

    # dataset creation and formating
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_pandas(processed_df)

    # TODO: truncation in effect so split longer speeches before this
    def tokenize_function(row):
        if cfg['model'] == 'speaker_prompt':
            return {**tokenizer('The following is a speech by '+ row['first_name'] + ' ' + row['last_name'] + '.' + row['speech'],truncation=True,max_length=cfg['max_seq_len'])}
        elif cfg['model'] == 'prefix_tuning':
            return {**tokenizer(row['speech'],truncation=True,max_length=cfg['max_seq_len'])}
        else:
            raise NotImplementedError()

    dataset = dataset.map(tokenize_function, batched=False) # change batched = True

    if cfg['model'] == 'speaker_prompt':
        included_cols = ['input_ids', 'attention_mask']
    elif cfg['model'] == 'prefix_tuning':
        included_cols = [cfg['encoded'],'input_ids', 'attention_mask']
    else:
        raise NotImplementedError()

    dataset.set_format(type='torch', columns=included_cols)

    # for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')

    dataloader = torch.utils.data.DataLoader(dataset,collate_fn=data_collator, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

    return dataloader, tokenizer

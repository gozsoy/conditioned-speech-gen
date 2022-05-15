import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import GPT2Tokenizer

def get_data(cfg, split=0):

    # read whole corpus
    processed_df = pd.read_csv(cfg['data_path'])

    # work on subset of speakers
    temp_bioguides = ['H001055','C001074']
    temp_df = processed_df[processed_df.bioguide_id.isin(temp_bioguides)]
    #temp_df = processed_df.iloc[:1280] #DELETE THIS LINE

    # dataset creation and formating
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_pandas(temp_df)

    # TODO: truncation in effect so split longer speeches before this
    # TODO: different prompt designs
    def tokenize_function(row):
        return {**tokenizer(row['first_name'] + ' ' + row['last_name'] + row['speech'],truncation=True,max_length=cfg['max_seq_len'])}

    dataset = dataset.map(tokenize_function, batched=False) # change batched = True
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')

    dataloader = torch.utils.data.DataLoader(dataset,collate_fn=data_collator, batch_size=cfg['batch_size'], shuffle=True)

    return dataloader, tokenizer
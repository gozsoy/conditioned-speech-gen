import re
import os
import glob
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score, roc_auc_score

# fix the seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

max_word_count = 120

# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained("m-newhauser/distilbert-political-tweets", cache_dir='/cluster/scratch/goezsoy/huggingface_cache')

net = AutoModelForSequenceClassification.from_pretrained("m-newhauser/distilbert-political-tweets", cache_dir='/cluster/scratch/goezsoy/huggingface_cache')


def tokenize_function(row):

    tokenizer_dict = tokenizer(row['speech'])
    tokenizer_dict['labels'] = row['label']

    return {**tokenizer_dict}


def prepare_dataloader(df, shuffle=True):

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(tokenize_function, batched=True)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')

    dataloader = torch.utils.data.DataLoader(dataset,collate_fn=data_collator, batch_size=16, shuffle=shuffle, drop_last=False)

    return dataloader


# load valid data for training on real examples
processed_df_valid = pd.read_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/processed_df_valid.csv')

# truncate speeches to max_word_count
processed_df_valid['speech'] = processed_df_valid['speech'].map(lambda row: ' '.join(row.split()[:max_word_count]))

# D -> 1, R -> 0
processed_df_valid['label'] = processed_df_valid['term_party'].apply(lambda row: 1 if row=='D' else 0)

processed_df_valid = processed_df_valid[['speech','label']]

test_dataloader = prepare_dataloader(processed_df_valid, shuffle=False)

net = net.to(device)

print('training done, starting test set evaluation.\n')

# evaluation
net.eval()
with torch.no_grad():

    batch_loss_array_test=[]
    
    batch_test_gt = []
    batch_test_preds = []
    batch_real_probs = []

    for _, test_batch_data in enumerate(test_dataloader): # loop over valid batches
        
        test_batch_data = test_batch_data.to(device)

        # forward pass
        test_outputs = net(**test_batch_data)
        test_loss = test_outputs.loss

        batch_test_gt += test_batch_data['labels'].tolist()
        batch_test_preds += torch.argmax(test_outputs.logits,axis=1).tolist()
        batch_real_probs += torch.nn.functional.softmax(test_outputs.logits, dim=1)[:,1].tolist()

        # save batch metrics
        detached_test_loss = test_loss.detach().cpu()
        batch_loss_array_test.append(detached_test_loss.item())

    test_loss = np.mean(batch_loss_array_test)

    test_acc = accuracy_score(batch_test_gt, batch_test_preds)
    test_f1 = f1_score(batch_test_gt, batch_test_preds)
    conf_matrix = confusion_matrix(batch_test_gt, batch_test_preds)
    auroc = roc_auc_score(batch_test_gt, batch_real_probs)
    auprc = average_precision_score(batch_test_gt, batch_real_probs)
    #tn, fp, fn, tp = confusion_matrix(batch_test_gt, batch_test_preds).ravel()
    #test_fpr = fp / (fp + tn)

    print(f'test_f1: {test_f1:.4f}, test_acc: {test_acc:.4f}, test_auroc: {auroc:.4f}, test_auprc: {auprc:.4f}')
    print('labeling -> 0: republican, 1: democrat')
    print(conf_matrix)

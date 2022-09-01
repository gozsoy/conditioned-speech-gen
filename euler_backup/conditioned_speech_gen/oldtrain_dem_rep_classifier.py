import re
import os
import glob
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
net = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


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

# train vs test split
X_train_valid, X_test, _, _ = train_test_split(processed_df_valid.index, processed_df_valid['label'], test_size=0.2, random_state=0, stratify=processed_df_valid['label'])

train_valid_df = processed_df_valid.iloc[X_train_valid].reset_index(drop=True)
test_df = processed_df_valid.iloc[X_test].reset_index(drop=True)

# train vs valid split
X_train, X_valid, _, _ = train_test_split(train_valid_df.index, train_valid_df['label'], test_size=0.1, random_state=0, stratify= train_valid_df['label'])

train_df = train_valid_df.iloc[X_train].reset_index(drop=True)
valid_df = train_valid_df.iloc[X_valid].reset_index(drop=True)


train_dataloader = prepare_dataloader(train_df)

valid_dataloader = prepare_dataloader(valid_df)

test_dataloader = prepare_dataloader(test_df, shuffle=False)

print('data loaded, starting model training.\n')


net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-5)

# zero the parameters' gradients
optimizer.zero_grad()

epochs = 5
for epoch in range(epochs):  # loop over dataset

    net.train()

    batch_loss_array=[]

    batch_train_gt = []
    batch_train_preds = []

    # training
    for batch_idx, batch_data in enumerate(train_dataloader): # loop over train batches
        
        batch_data = batch_data.to(device)

        # forward pass
        outputs = net(**batch_data)
        loss = outputs.loss

        batch_train_gt += batch_data['labels'].tolist()
        batch_train_preds += torch.argmax(outputs.logits,axis=1).tolist()

        # backpropagation
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.zero_grad()

        # save batch metrics
        detached_loss = loss.detach().cpu()
        batch_loss_array.append(detached_loss.item())

    # validation
    net.eval()
    with torch.no_grad():

        batch_loss_array_valid=[]

        batch_valid_gt = []
        batch_valid_preds = []

        for _, valid_batch_data in enumerate(valid_dataloader): # loop over valid batches
            
            valid_batch_data = valid_batch_data.to(device)

            # forward pass
            val_outputs = net(**valid_batch_data)
            val_loss = val_outputs.loss

            batch_valid_gt += valid_batch_data['labels'].tolist()
            batch_valid_preds += torch.argmax(val_outputs.logits,axis=1).tolist()

            # save batch metrics
            detached_val_loss = val_loss.detach().cpu()
            batch_loss_array_valid.append(detached_val_loss.item())


    # display metrics at end of epoch
    epoch_train_loss, epoch_val_loss = np.mean(batch_loss_array), np.mean(batch_loss_array_valid)

    train_acc = accuracy_score(batch_train_gt, batch_train_preds)
    valid_acc = accuracy_score(batch_valid_gt, batch_valid_preds)

    print(f'epoch: {epoch+1} / {epochs}, train_loss: {epoch_train_loss:.4f}, val_loss: {epoch_val_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {valid_acc:.4f}')

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

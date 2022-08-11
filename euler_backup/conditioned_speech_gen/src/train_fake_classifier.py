import pandas as pd
import torch
import torch.optim as optim
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
net = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


def tokenize_function(row):

    tokenizer_dict = tokenizer(row['speech'])
    tokenizer_dict['labels'] = row['label']

    return {**tokenizer_dict}


def prepare_dataloader(df):

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(tokenize_function, batched=True)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')

    dataloader = torch.utils.data.DataLoader(dataset,collate_fn=data_collator, batch_size=16, shuffle=True, drop_last=False)

    return dataloader


real_fake_df = pd.read_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/real_fake_df.csv')

real_fake_df['speech'] = real_fake_df['speech'].map(lambda row: row.lower())

# train vs test split
X_train_valid, X_test, _, _ = train_test_split(real_fake_df.index, real_fake_df['label'], test_size=0.2, random_state=0, stratify=real_fake_df['label'])

real_fake_train_valid_df = real_fake_df.iloc[X_train_valid].reset_index(drop=True)
real_fake_test_df = real_fake_df.iloc[X_test].reset_index(drop=True)

# train vs valid split
X_train, X_valid, _, _ = train_test_split(real_fake_train_valid_df.index, real_fake_train_valid_df['label'], test_size=0.1, random_state=0, stratify= real_fake_train_valid_df['label'])

real_fake_train_df = real_fake_train_valid_df.iloc[X_train].reset_index(drop=True)
real_fake_valid_df = real_fake_train_valid_df.iloc[X_valid].reset_index(drop=True)


train_dataloader = prepare_dataloader(real_fake_train_df)

valid_dataloader = prepare_dataloader(real_fake_valid_df)

test_dataloader = prepare_dataloader(real_fake_test_df)

print('data loaded, starting model training.\n')


net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-5)

# zero the parameters' gradients
optimizer.zero_grad()

epochs = 10
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

    for _, test_batch_data in enumerate(test_dataloader): # loop over valid batches
        
        test_batch_data = test_batch_data.to(device)

        # forward pass
        test_outputs = net(**test_batch_data)
        test_loss = test_outputs.loss

        batch_test_gt += test_batch_data['labels'].tolist()
        batch_test_preds += torch.argmax(test_outputs.logits,axis=1).tolist()

        # save batch metrics
        detached_test_loss = test_loss.detach().cpu()
        batch_loss_array_test.append(detached_test_loss.item())

    test_loss = np.mean(batch_loss_array_test)

    test_acc = accuracy_score(batch_test_gt, batch_test_preds)
    conf_matrix = confusion_matrix(batch_test_gt, batch_test_preds)

    print(f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} \n')

    print('labeling -> 0: fake, 1: real')
    print(conf_matrix)
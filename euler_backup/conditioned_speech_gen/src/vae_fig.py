import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import DistilBertModel, BertLMHeadModel, BertTokenizerFast
from transformers import DistilBertConfig, BertConfig

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

processed_df = pd.read_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/processed_df_valid.csv')

processed_df = processed_df.iloc[:500]

# dataset creation and formating
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

dataset = Dataset.from_pandas(processed_df)

# TODO: truncation in effect so split longer speeches before this
def tokenize_function(row):
    return {**tokenizer(row['speech'],truncation=True,max_length=256)}

dataset = dataset.map(tokenize_function, batched=False) # change batched = True

included_cols = ['input_ids', 'attention_mask']
dataset.set_format(type='torch', columns=included_cols)

# for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')

dataloader = torch.utils.data.DataLoader(dataset,collate_fn=data_collator, batch_size=16, shuffle=False, drop_last=False)


class BERT_VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        encoder_config = DistilBertConfig.from_pretrained('distilbert-base-cased')
        encoder_config.gradient_checkpointing = True

        decoder_config = BertConfig.from_pretrained('bert-base-cased')
        decoder_config.gradient_checkpointing = True
        
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-cased', config=encoder_config)
        self.decoder = BertLMHeadModel.from_pretrained('bert-base-cased', config=decoder_config)

        self.mu_linear = nn.Linear(self.encoder.config.hidden_size,cfg['latent_dim'])
        self.rho_linear = nn.Linear(self.encoder.config.hidden_size,cfg['latent_dim'])

        self.distributor = nn.Linear(cfg['latent_dim'],self.encoder.config.hidden_size)

    def forward(self, inputs):
        
        device = next(self.encoder.parameters()).device

        enc_output = self.encoder(**inputs)['last_hidden_state']

        cls_tokens = enc_output[:,0,:]

        z_mu = self.mu_linear(cls_tokens)
        z_rho = self.rho_linear(cls_tokens)
        
        epsilon = torch.normal(0.0,1.0,size=z_mu.shape).to(device)

        z = z_mu + F.softplus(z_rho) * epsilon

        dec_input = torch.zeros_like(enc_output).to(device)
        dec_input[:,0,:] = self.distributor(z)
        dec_input[:,1:,:] = enc_output[:,1:,:]
        
        # remove positional encodings because decoder will add them during forward pass again
        dec_input = dec_input - self.encoder.embeddings.position_embeddings(torch.arange(start=0,end=dec_input.shape[1]).to(device))

        outputs = self.decoder(inputs_embeds=dec_input,attention_mask=inputs['attention_mask'])
        
        return z_mu, z_rho, outputs.logits



cfg = {'latent_dim':200}

model = BERT_VAE(cfg).to(device)

checkpoint = torch.load('/cluster/scratch/goezsoy/nlp_lss_checkpoints/9june_bert_vae_latent200_kl100_maxseqlen256_batch8_8_lr1e5_epoch6.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


z_mu_array = None

with torch.no_grad():

    for b in dataloader:
        
        b = b.to(device)
        enc_output = model.encoder(**b)['last_hidden_state']

        cls_tokens = enc_output[:,0,:]

        z_mu = model.mu_linear(cls_tokens)

        if z_mu_array is None:
            z_mu_array = z_mu.cpu().numpy()
        else:
            z_mu_array = np.concatenate((z_mu_array,z_mu.cpu().numpy()),axis=0)




X_embedded = TSNE(n_components=2).fit_transform(z_mu_array)

scatter_x = X_embedded[:,0]
scatter_y = X_embedded[:,1]

fig, ax = plt.subplots(figsize=(15,15))

ax.scatter(scatter_x, scatter_y)

for i in range(500):
    ax.annotate(i, (scatter_x[i], scatter_y[i]))

plt.savefig('9june_bert_vae_latent200_kl100_maxseqlen256_batch8_8_lr1e5_epoch6.png')
plt.show()
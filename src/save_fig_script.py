from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoForCausalLM

import torch
import torch.nn as nn


class PrefixNet(nn.Module):
    def __init__(self, cfg):
        super(PrefixNet, self).__init__()

        self.gpt_neo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

        self.batch_size = cfg['batch_size']
        self.prefix_len = cfg['prefix_len']
        self.token_size = self.gpt_neo.config.hidden_size # change this to config.hidden_size
        self.embed_size_per_token = cfg['embed_size_per_token']
        self.speaker_size = cfg['speaker_size']
        
        # freeze gpt layers
        if cfg['freeze_gpt']:
            for param in self.gpt_neo.parameters():
                param.requires_grad = False

        self.embedding_layer = nn.Embedding(num_embeddings=self.speaker_size, embedding_dim=self.embed_size_per_token * self.prefix_len)
        self.linear_layer = nn.Linear(in_features=self.embed_size_per_token, out_features=self.token_size)

    def forward(self, just_wte, speaker_id):

        device = next(self.gpt_neo.parameters()).device
        
        prefix_tensor = torch.zeros(size=(self.batch_size, self.prefix_len, self.token_size)).to(device)

        speaker_embs = self.embedding_layer(speaker_id)
        
        for i in range(self.prefix_len):
            prefix_tensor[:,i,:] = self.linear_layer(speaker_embs[:,i*self.embed_size_per_token:(i+1)*self.embed_size_per_token])

        prefix_cat_token_embs = torch.cat((prefix_tensor,just_wte), dim = 1)

        output = self.gpt_neo(inputs_embeds=prefix_cat_token_embs)
        
        return output.logits


if __name__ == '__main__':

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    cfg= {'batch_size':8,'prefix_len':3,'embed_size_per_token':100,'speaker_size':20, 'freeze_gpt':False}

    model = PrefixNet(cfg)

    checkpoint = torch.load('../checkpoints/23may_prefix_tuning_prlen3_embsize100_speaksize20_maxseqlen256_batch8_8_epoch90.pt',map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X = model.embedding_layer.weight.detach().numpy()

    X_embedded_tsne = TSNE(n_components=2).fit_transform(X)
    X_embedded_pca = PCA(n_components=2).fit_transform(X)
    X_embedded_mds = MDS(n_components=2).fit_transform(X)
    X_embedded_isomap = Isomap(n_components=2).fit_transform(X)

    speaker_labels = np.arange(0,20)
    processed_df = pd.read_csv('../data/processed_df.csv')
    meta_df= processed_df[processed_df.encoded_bioguide_ids.isin(speaker_labels)].drop_duplicates(subset='encoded_bioguide_ids')

    group = meta_df.term_party.values
    cdict = {'Democrat': 'blue', 'Republican': 'red'}

    fig, axs = plt.subplots(2,2)

    for g in np.unique(group):
        ix = np.where(group == g)
        axs[0,0].scatter(X_embedded_tsne[ix,0], X_embedded_tsne[ix,1], c = cdict[g], label = g, s = 100)

    for i, txt in enumerate(list(speaker_labels)):
        axs[0,0].annotate(txt, (X_embedded_tsne[i,0], X_embedded_tsne[i,1]))
    axs[0,0].legend()
    axs[0,0].set_title('tsne')

    for g in np.unique(group):
        ix = np.where(group == g)
        axs[0,1].scatter(X_embedded_pca[ix,0], X_embedded_pca[ix,1], c = cdict[g], label = g, s = 100)

    for i, txt in enumerate(list(speaker_labels)):
        axs[0,1].annotate(txt, (X_embedded_pca[i,0], X_embedded_pca[i,1]))
    axs[0,1].legend()
    axs[0,1].set_title('pca')

    for g in np.unique(group):
        ix = np.where(group == g)
        axs[1,0].scatter(X_embedded_mds[ix,0], X_embedded_mds[ix,1], c = cdict[g], label = g, s = 100)

    for i, txt in enumerate(list(speaker_labels)):
        axs[1,0].annotate(txt, (X_embedded_mds[i,0], X_embedded_mds[i,1]))
    axs[1,0].legend()
    axs[1,0].set_title('mds')

    for g in np.unique(group):
        ix = np.where(group == g)
        axs[1,1].scatter(X_embedded_isomap[ix,0], X_embedded_isomap[ix,1], c = cdict[g], label = g, s = 100)

    for i, txt in enumerate(list(speaker_labels)):
        axs[1,1].annotate(txt, (X_embedded_isomap[i,0], X_embedded_isomap[i,1]))
    axs[1,1].legend()
    axs[1,1].set_title('isomap')

    plt.savefig('23may_prefix_tuning_prlen3_embsize100_speaksize20_maxseqlen256_batch8_8_epoch90.png')
    plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers import DistilBertModel, BertLMHeadModel
from transformers import DistilBertConfig, BertConfig


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
        config.gradient_checkpointing = True
        config.use_cache = False

        self.gpt_neo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", config=config)

    def forward(self, inputs):
        
        outputs = self.gpt_neo(**inputs, labels=inputs.input_ids)
        
        return outputs.loss, outputs.logits


# do not forget to add gradient checkpointing
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

        # TODO: pass attention mask to below as well!!
        output = self.gpt_neo(inputs_embeds=prefix_cat_token_embs)
        
        return output.logits


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
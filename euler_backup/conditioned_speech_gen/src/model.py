import torch
import torch.nn as nn
from transformers import GPTNeoConfig, GPTNeoForCausalLM


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

        output = self.gpt_neo(inputs_embeds=prefix_cat_token_embs)
        
        return output.logits
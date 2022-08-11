import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers import DistilBertModel, BertLMHeadModel
from transformers import DistilBertConfig, BertConfig
from transformers import GPT2Config, GPT2LMHeadModel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
        config = GPT2Config.from_pretrained("gpt2-medium")
        config.gradient_checkpointing = True
        config.use_cache = False

        #self.gpt_neo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", config=config)
        self.gpt_neo = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=config)

    def forward(self, inputs):
        
        outputs = self.gpt_neo(**inputs, labels=inputs.input_ids)
        
        return outputs.loss, outputs.logits
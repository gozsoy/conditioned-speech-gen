import torch
import torch.nn as nn
from transformers import GPTNeoForCausalLM


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gpt_neo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    def forward(self, inputs):
        
        outputs = self.gpt_neo(**inputs, labels=inputs.input_ids)
        
        return outputs.loss, outputs.logits
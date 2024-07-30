from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # This statement assures that the embedding size is divisible by
        # the attention heads in the attention layer to prevent loss of information
        # while diminishing the embedding dimensions into number of attention heads.
        assert config.n_embed % config.n_head == 0

        #get the key query and value for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)

        #outpubt projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        #regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)


        #not really a 'bais' , more of a mask but following OpenAI/HF naming 
        #register_buffer(name, tensor, persistent=True)[source] Add a buffer 
        #to the module. This is typically used to register a buffer that should 
        #not to be considered a model parameter
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        #B -> Batch size, 
        #T -> Time-steps, 
        #C -> channels
        #calculate query, key, values for all heads in batch and move head 
        #forward to be the batch_size
        #nh is the "number of heads", hs is "head_size", and C (number of Channels) = nh*hs
        #eg: in GPT2, n_head=12, hs=64, so nh*hs=784 channels in transformer 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            
        """Standard Implementation of Attention
        #attention (materializes the larget (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, 1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        """

        """Flash Attention."""
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.2)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all heads outputs side by side
      
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
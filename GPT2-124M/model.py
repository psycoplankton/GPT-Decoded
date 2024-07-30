from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from attention import CasualSelfAttention


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        """A smooth version of RELU"""
        self.gelu = nn.GELU()
        
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        #add a flag variable 
        self.c_proj.NONOGPT_SCALE_INIT = 1 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    """Structure of the architecture
    1. inputs are first layer normalized
    2. Passed through the "Casual Self Attention" pipeline
    3. Layer Normalized again
    4. Lastly fed to an MLP to generate the output."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

"""This method automatically adds the __init__() and __repr__() constructor methods"""
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.2
    bias: bool = True

class GPT(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        """all of the attention heads and encodings wraped in a method"""
        self.transformer = nn.ModuleDict(dict(
            # Text encodings
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            
            #Position encodings
            wpe = nn.Embedding(config.block_size, config.n_embed),

            drop = nn.Dropout(config.dropout),
            
            #Wrapper for all the Modules
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            #Layer Normalization at the end of the decoder block
            #This is one of the new modifications in GPT2 comapred to the 
            #original transformer architecture
            ln_f = nn.LayerNorm(config.n_embed, bias=config.bias),
        ))
        

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight 

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *=(2*self.config.n_layer) ** 0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        #idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequnece of length {T}, block size is only {self.config.block_size}"

        #forward the token and position embeddings
        pos = torch.arange(T ,dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        #forward the blocks of the transformers
        for block in self.transformer.h:
            x = block(x)
        
        #forward
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pre-trained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pre-trained gpt: %s" % model_type)

        #n_layer, n_head, and n_embed are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all the prameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        #basically the openai checkpoints use a Conv1D module but we only want to use
        # the vanilla MLP, this means we have to transpose the weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

            return model
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        """According to the training of GPT3, a weight decay of 0.1 is applied to all the weights
        which are a part of 2 dimensional weight matrix. For example, say the token embeddings, positonal embeddings
        weights of the attention layer, all of these decay but the biases in feed-forward layer, layer-normalization
        do not decay."""
        #start with all of the candidate parameters (that require grad)

        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}

        #create optim groups. All parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
         
        decay_params = [p for n, p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params':decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_decay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed paramter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non_decayed parameter tensors: {len(nodecay_params)}, with {num_decay_params:,} params")

        #create AdamW optimizer and use the fused version if it is not available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps = 1e-8)
        
        return optimizer 
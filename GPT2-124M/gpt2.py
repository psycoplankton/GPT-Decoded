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
### -----------------------------------------------------------------------------###
# DATALOADER WRAPPER FOR CREATING BATCHES
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        #at init load tokens from disk ans store them in memory
        """Read the input data"""
        with open('input.txt', 'r') as f:
            text = f.read()

        """Get the gpt2 encodings from tiktoken and encode the entire dataset"""
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // self.B*self.T} batches")

        #state
        self.current_positon = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_positon : self.current_positon + B*T +1]
        buf = buf.to(device)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        #advance the position in the tensor
        self.current_positon+= B*T
        if(self.current_positon + (B*T +1) > len(self.tokens)):
            self.current_positon = 0
        return x, y


### -------------------------------------------------------------------------###
# TRAINING 
"""Set the device to cuda or cpu"""
device='cpu'
if torch.cuda.is_available():
    device='cuda'

"""Read the input data"""
with open('input.txt', 'r') as f:
    text = f.read()

"""Get the gpt2 encodings from tiktoken and encode the entire dataset"""
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)

epochs = 30
num_return_sequences = 5
max_length = 30

B = 32 #micro batch_size
T = 128 #sequence length


"""we are going to map every encoding onto it's next one as an (input, label) pair"""
train_loader = DataLoaderLite(B=B, T=T)

#shakespeare = GPT.from_pretrained('gpt2')
shakespeare = GPT(GPTConfig(vocab_size=50306)) 
print("compiled successfully")
shakespeare.to(device)

max_lr = 0.001
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 10000
#-------------------------------------------------------------------------------------------#
#COSINE LEARNING RATE DECAY 
def cosine_lr_schedular(it):
    # 1) linear warmup for warmup-steps iterations.
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    # 2) if it>max_steps return the minimum lr
    if it> max_steps:
        return min_lr

    # in between , apply the cosine decay for the learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr) 
#-------------------------------------------------------------------------------------------#
# OPTIMZATION LOOP
"""Pytorch compiler behaves like a gcc compiler where it sees all of the code
and already knows what operations are going to be performed, unlike the intepreter
which makes the process faster."""
shakespeare = torch.compile(shakespeare)

"""Optimizer parameters according to the GPT3 paper"""
optimizer = shakespeare.configure_optimizers(weight_decay=0.1, learning_rate=min_lr, betas=(0.9, 0.95), device=device)
for i in range(max_steps):
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device ):
        logits, loss = model(x, y)     
    loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(shakespeare.parameters(), 1.0)
    
    #use the cosine learning rate decay to the get lr for i'th step
    #change the optimizer lr to this lr
    #take the optimizer step
    lr = cosine_lr_schedular(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
     
    print(f"loss -> {loss.item()} at iteration -> {i+1}") 
###-------------------------------------------------------------------------------------------------###

#prefix tokens 
import tiktoken

#get the encodings for gpt2
enc = tiktoken.get_encoding('gpt2')

tokens = enc.encode("Hello!, I'm a langauge model.")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

#function to generate or sample from the model distribution
#set seed = 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

###-----------------------------------------------------------------------------------###
#SAMPLING LOOP

#generate until the length of the sentence is equal to max_length 
while x.size(1) < max_length:

    #mention beforehand that no gradient is going to be called on this
    # block of code, to speed up the process.
    with torch.no_grad():
        logits, _ = shakespeare(x) #shape = (B, T, vocab_size)
        #take the logits at the last position
        logits = logits[:, -1, :] #shape = (B, vocab_size)
        #get the probabilities
        probs = F.softmax(logits, dim=-1)
        #do a top- sampling of 50 (hugging-face default)
        #topk_probs here becomes (5, 50), topk_indices is(5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        #select a token from the topk probabilities
        ix = torch.multinomial(topk_probs, 1) #shape (B, 1)
        #gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #shape (B, 1)
        #append to the sequence
        x = torch.cat([x, xcol], dim=1)
###-----------------------------------------------------------------------------###
#print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
         

 

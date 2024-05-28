import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from attention import Block
class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block(config.embed, config.n_head) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, vocab_size)

        #better init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx : torch.Tensor, targets = None) -> None:
        #B = Batch, T = Time; basically the idx, C = Channels; dimensions of embeddings
        B, T = idx.shape

        tok_emb = self.token_embeddings_table(idx) # shape (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = (F.cross_entropy(logits, targets))

        return logits, loss
    
    def generate(self, idx : torch.Tensor, max_new_tokens : int) -> torch.Tensor:
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
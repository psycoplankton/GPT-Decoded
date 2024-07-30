from typing import List, Union, Dict, Callable
import torch
import config
def get_chars(text : str) -> List:
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("The characters are: \n"''.join(chars))
    print("The vocabulary size is: ", vocab_size)
    return chars

def get_tokens(chars : List) -> Union[Dict, Dict, ]:
    #create a mapping from characters to integers
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    return stoi, itos
     

def train_test_split(data):
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data

def get_batch(train_data : torch.Tensor, val_data : torch.Tensor, mode : str):
    data = train_data if mode == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model : torch.nn.Module, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

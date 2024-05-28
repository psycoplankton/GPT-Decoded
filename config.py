import torch


# HYPERPARAMETERS

batch_size = 64
n_embed = 384
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256
eval_iters = 200
n_layer = 6
n_head = 6
dropout = 0.2

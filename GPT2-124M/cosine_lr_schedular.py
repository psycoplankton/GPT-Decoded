import math

max_lr = 0.001
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 10000

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
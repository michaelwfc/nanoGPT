"""build GPT from scratch

from [Andrej Karpathy- Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

block_size = 8
batch_size = 32
max_iterations = 3000
eval_interval = 300
eval_iters = 1000
learning_rate = 1e-2

device = "cuda" if torch.cuda.is_available() else 'cpu'

# read it in to inspect it
with open('./data/shakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text))
print(f"data.shape={data.shape}")
print(f"data.type={data.dtype}")


# Let's now split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split="train", batch_size=4):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[x:x+block_size] for x in ixs])
    y = torch.stack([data[x+1: x+block_size + 1] for x in ixs])
    x, y = x.to(device), y.to(device)
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=None):
        super().__init__()
        embedding_size = embedding_size if embedding_size else vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        # self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)
        # logits = self.linear(logits)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx = (B,T)
        for _ in range(max_new_token):
            logits, loss = self(idx)  # logits=(B,T,C)
            logits_last = logits[:, -1, :]  # (B,C)
            probs = F.softmax(logits_last, dim=1)  # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for step in range(max_iterations):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train", batch_size=batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(f"loss={loss.item()}")


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_idx = model.generate(idx, max_new_token=1000)[0].tolist()
# print(f"generated_idx={generated_idx}")

generated_tokens = decode(generated_idx)
print(f"generated_tokens={generated_tokens}")

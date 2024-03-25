"""build GPT from scratch

from [Andrej Karpathy- Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

BLOCK_SIZE = 8
batch_size = 32

eval_interval = 300
eval_iters = 1000


EMBEDDING_SIZE = 64


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


def get_batch(split="train", batch_size=4, block_size=BLOCK_SIZE):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,))  # (B,)
    x = torch.stack([data[x:x+block_size] for x in ixs])        # (B, block_size)
    y = torch.stack([data[x+1: x+block_size + 1] for x in ixs])  # (B, block_size)
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    def __init__(self, embedding_size, block_size, head_size, dropout_rate, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.head_size = head_size
        self.device = device
        self.key = nn.Linear(embedding_size, head_size, bias=False, device=device)
        self.query = nn.Linear(embedding_size, head_size, bias=False, device=device)
        self.value = nn.Linear(embedding_size, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x = (B,T,C)
        (B, T, C) = x.shape
        k = self.key(x)  # (B,T,H)
        q = self.query(x)  # (B,T,H)
        v = self.value(x)  # (B,T,H)
        wei = k @ q.transpose(-2, -1) * self.head_size**(-0.5)  # (B,T,T)
        # tril = torch.tril(torch.ones(T, T))
        # when generate the T is change from 1 to max_generate_token
        wei.masked_fill_(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # add dropout for wei
        wei = self.dropout(wei)
        out = wei @ v  # (B,T,H)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, block_size, num_head, head_size, dropout_rate, device):
        super().__init__()
        self.num_head = num_head
        self.head_size = head_size
        self.device = device
        self.heads = nn.ModuleList([Head(embedding_size=embedding_size,
                                         block_size=block_size,
                                         head_size=head_size, dropout_rate=dropout_rate,
                                         device=device)
                                    for _ in range(self.num_head)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        # add dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # add projection
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*4),
            nn.ReLU(),
            # add projection
            nn.Linear(embedding_size*4, embedding_size),
            # add dropout
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # x = (B, T, C)
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embedding_size, block_size, num_head, dropout_rate, device) -> None:
        super().__init__()
        head_size = embedding_size // num_head
        self.self_attention = MultiHeadAttention(embedding_size=embedding_size, block_size=block_size,
                                                 num_head=num_head, head_size=head_size, dropout_rate=dropout_rate,
                                                 device=device)
        self.ffwd = FeedForward(embedding_size=embedding_size, dropout_rate=dropout_rate)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # add prenormalization for self attention
        x = self.ln1(x)
        # add self_attention = residule connection
        x = x + self.self_attention(x)

        # add prenormalization for feed forward
        x = self.ln2(x)
        # add feed forward +  residule connection
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self,  vocab_size, embedding_size, block_size,
                 num_head=4, n_layer=4, dropout_rate=0.2, device=None):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding_table = nn.Embedding(block_size, embedding_size)

        # self.self_attention_head = Head(embedding_size=embedding_size, block_size=block_size, head_size=embedding_size, device=device)

        # head_size = embedding_size//num_head
        # self.self_attention_head = MultiHeadAttention(embedding_size=embedding_size, block_size=block_size,
        #                                               num_head=num_head, head_size=head_size,
        #                                               device=device)
        # self.ffwd = FeedForward(embedding_size=embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size=embedding_size, block_size=block_size, num_head=num_head,
                                    dropout_rate=dropout_rate, device=device) for _ in range(n_layer)])
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integes
        B, T = idx.shape
        tok_embedding = self.token_embedding_table(idx)  # (B,T,C)

        pos = torch.arange(T, device=self.device)  # (T,)
        pos_embedding = self.pos_embedding_table(pos)  # (T,C)
        x = tok_embedding + pos_embedding  # (B,T,C)

        # apply one head or multihead of self-attention
        # x = self.self_attention_head(x)  # (B,T,C)
        # apply ffwd
        # x = self.ffwd(x)  # (B,T,C)

        # apply blockes of multihead of self-attention
        x = self.blocks(x)  # (B,T,C)

        logits = self.linear(x)

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
            # crop idx to th last block size tokens
            idx_cond = idx[:, -self.block_size:]  # (B, block_size)
            logits, loss = self(idx_cond)  # logits=(B,T,C)
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
            x, y = get_batch(split, batch_size, block_size=BLOCK_SIZE)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    max_iterations = 5000
    learning_rate = 1e-4
    dropout_rate = 0.2
    num_head = 4
    model = BigramLanguageModel(vocab_size=vocab_size, embedding_size=EMBEDDING_SIZE,
                                block_size=BLOCK_SIZE, num_head=num_head, dropout_rate=dropout_rate,
                                device=device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iterations):
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", batch_size=batch_size, block_size=BLOCK_SIZE)
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

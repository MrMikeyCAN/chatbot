import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open("/Users/mertcan/Desktop/Chatbot/input2.txt", "r", encoding="utf-8") as f:
    text = f.read()

torch.manual_seed(42)
words = text.split()
vocab_size = len(set(words))
# Create a mapping from words to integers
sentences_to_indices = {w: i for i, w in enumerate(set(words))}
indices_to_sentences = {i: w for i, w in enumerate(set(words))}


def encode(sentence):
    return [sentences_to_indices[word] for word in sentence.split()]


def decode(indices):
    return " ".join(indices_to_sentences[i] for i in indices)


# Hyper params
n_embd = 8
n_head = 6
n_layer = 20
dropout = 0.3
batch_size = 4
block_size = 4
decoder = decode
encoder = encode
device = device

# Training params
learning_rate = 3e-4
device = device
max_iters = 5000
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[: n]
val_data = data[n:]
checkpoint = 100
eval_interval = 1
eval_iters = 100
graficate = False
text = text


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    _data = train_data if split == "train" else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix]).to(device)
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        b, t, c = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
                q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
        )
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_size = block_size
        self.positional_encoding = self._generate_positional_encoding()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(
            self.block_size, n_embd
        )
        self.decoder = decoder
        self.blocks = nn.Sequential(
            *[
                Block(
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self.__init__weights)

    @staticmethod
    def _generate_positional_encoding() -> torch.Tensor:
        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
        positional_encoding = torch.zeros(block_size, n_embd)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(positional_encoding, requires_grad=True)

    @staticmethod
    def __init__weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        pos_emb = self.positional_encoding[:t, :].unsqueeze(0)  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            if "<END>" in self.decoder(idx[0].tolist()):
                print("-------------FINISH-------------")
                break
        return idx


model = GPTLanguageModel().to(device)
m = model.to(device)
print("Device of model:", next(model.parameters()).device)
print("Train parameters device:", device)
print("Hyper parameters device:", device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def generate_text(
        context: str,
        max_new_tokens: int = 500,
):
    generated_text = ""
    context = torch.tensor(encoder(context), device=device)[
              None, :
              ]
    generated_text += decoder(
        m.generate(context, max_new_tokens)[0].tolist()
    )
    return generated_text


def train():
    model.train()
    dir_name = f"/Users/mertcan/Desktop/Chatbot/checkpoint/{time.time()}"
    train_losses = []
    val_losses = []
    iter_values = []
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_iters
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
    for iters in range(max_iters):
        start_time = time.time()

        # every once in a while evaluate the loss on train and val sets
        if iters % eval_interval == 0 or iters == max_iters - 1:
            losses = estimate_loss()
            train_loss = losses["train"]
            val_loss = losses["val"]
            print(
                f"step {iters}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            iter_values.append(iter)
            print(generate_text("my name is", 5))

            if iter != 0 and checkpoint != 0 and iters % checkpoint == 0:
                path_name = dir_name + f"/checkpoint:{iters}.h5"
                torch.save(
                    model.state_dict(),
                    path_name,
                )
                print("checkpoint successfully saved")

            print("------------------------------------")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time}")

    # Plot losses
    if graficate:
        plt.plot(iter_values, train_losses, label="Train Loss")
        plt.plot(iter_values, val_losses, label="Val Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()
        plt.show()
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved successfully")

# train()

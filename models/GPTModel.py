import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class Hyperparameters:
    def __init__(
        self,
        bath_size,
        block_size,
        max_iters,
        eval_interval,
        learning_rate,
        device,
        eval_iters,
        n_embeaded,
        n_head,
        n_layer,
        dropout,
    ):
        self.batch_size = bath_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters
        self.n_embd = n_embeaded
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
hyperparams = Hyperparameters(8, 64, 5000, 1, 3e-4, device, 200, 384, 6, 6, 0.2)
# ------------


torch.manual_seed(1337)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split, hyperparams: Hyperparameters):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - hyperparams.block_size, (hyperparams.batch_size,))
    x = torch.stack([data[i : i + hyperparams.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + hyperparams.block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(hyperparams: Hyperparameters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(hyperparams.eval_iters)
        for k in range(hyperparams.eval_iters):
            X, Y = get_batch(split, hyperparams)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, block_size, dropout, n_embd, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_embd, dropout, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    dropout=dropout,
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                )
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

    def __init__(self, n_embd, dropout):
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

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            block_size=block_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, hyperparams: Hyperparameters):
        super().__init__()
        self.block_size = hyperparams.block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, hyperparams.n_embd)
        self.position_embedding_table = nn.Embedding(
            self.block_size, hyperparams.n_embd
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=hyperparams.n_embd,
                    n_head=hyperparams.n_head,
                    block_size=hyperparams.block_size,
                    dropout=hyperparams.dropout,
                )
                for _ in range(hyperparams.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(hyperparams.n_embd)  # final layer norm
        self.lm_head = nn.Linear(hyperparams.n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
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
        return idx


model = GPTLanguageModel(hyperparams)
m = model.to(device)

print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")


# generate from the model
def Generate_Text(
    context: str,
    max_new_tokens: int = 500,
):
    context = torch.tensor(encode(context), device=device)[None, :]
    print(decode(m.generate(context, max_new_tokens)[0].tolist()))


# create a PyTorch optimizer
def trainer(
    visualization: bool,
    hyperparams: Hyperparameters,
    checkpoints: int = 0,
    max_new_tokens: int = 500,
):
    train_losses = []
    val_losses = []
    iter_values = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)
    eval_interval = hyperparams.eval_interval
    max_iters = hyperparams.max_iters
    for iter in range(hyperparams.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(hyperparams)
            train_loss = losses["train"]
            val_loss = losses["val"]
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            iter_values.append(iter)
            Generate_Text("Hello world!", max_new_tokens)

            if checkpoints != 0 and iter % checkpoints == 0:
                torch.save(model.state_dict(), f"chechpoint/checkpoint:{iter}.pkl")
                print("checkpoint successfuly saved")

            print("------------------------------------")

        # sample a batch of data
        xb, yb = get_batch("train", hyperparams)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Plot losses
    if visualization:
        plt.plot(iter_values, train_losses, label="Train Loss")
        plt.plot(iter_values, val_losses, label="Val Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()
        plt.show()
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved successfully")


trainer(hyperparams=hyperparams, visualization=True, max_new_tokens=10, checkpoints=100)


# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

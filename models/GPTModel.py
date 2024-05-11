import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math
import os
import time
from utils import save_to_csv

device = "cuda" if torch.cuda.is_available() else "cpu"


# ! Hyper parametreler model için gerekli tüm parametreleri içerir ve de bir çok gerekli kodu
class Hyperparameters:
    def __init__(
            self,
            batch_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
            dropout,
            vocab_size,
            encoder,
            decoder,
            device,
    ):
        # * Batch size
        self.batch_size = batch_size
        # * Block size
        self.block_size = block_size
        # * Gömülü katman sayısı
        self.n_embd = n_embd
        self.n_head = n_head
        # * Katman sayısı (Kaç farklı sonuç sallayacağını seçer)
        self.n_layer = n_layer
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # hyperparameters


# ------------

torch.manual_seed(42)


# ! Eğitim için gerekli tüm kodları içerir
class TrainParameters:
    def __init__(
            self,
            text,
            max_iters,
            eval_interval,
            learning_rate,
            device,
            eval_iters,
            checkpoint: int,
            visualate: bool,
            encoder,
            decoder,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.data = torch.tensor(self.encoder(text), dtype=torch.long)
        self.n = int(0.9 * len(self.data))
        self.train_data = self.data[: self.n]
        self.val_data = self.data[self.n:]
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters
        self.checkpoint = checkpoint
        self.visualate = visualate


# data loading
def get_batch(split, hyperparams: Hyperparameters, trainParameters: TrainParameters):
    # generate a small batch of data of inputs x and targets y
    data = trainParameters.train_data if split == "train" else trainParameters.val_data
    ix = torch.randint(len(data) - hyperparams.block_size, (hyperparams.batch_size,))
    x = torch.stack([data[i: i + hyperparams.block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + hyperparams.block_size + 1] for i in ix]).to(device)
    x, y = x.to(trainParameters.device), y.to(trainParameters.device)
    return x, y


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


class PositionalEncoding:
    def __init__(self, block_size, n_embd):
        self.block_size = block_size
        self.n_embd = n_embd

    def _generate_positional_encoding(block_size, n_embd):
        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
        positional_encoding = torch.zeros(block_size, n_embd)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(positional_encoding, requires_grad=True)


class GPTLanguageModel(nn.Module):

    def __init__(self, hyperparams: Hyperparameters):
        super().__init__()
        self.block_size = hyperparams.block_size
        self.positional_encoding = PositionalEncoding._generate_positional_encoding(
            block_size=hyperparams.block_size, n_embd=hyperparams.n_embd
        )
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            hyperparams.vocab_size, hyperparams.n_embd
        )
        self.position_embedding_table = nn.Embedding(
            self.block_size, hyperparams.n_embd
        )
        self.decoder = hyperparams.decoder
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
        self.lm_head = nn.Linear(hyperparams.n_embd, hyperparams.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self.__init__weights)

    def __init__weights(self, module):
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

        pos_emb = self.positional_encoding[:T, :].unsqueeze(0)  # (T,C)

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


# ! Model ile ilgili tüm parametreler
class ModelFuncs:
    def __init__(
            self,
            model: GPTLanguageModel,
            hyperparams: Hyperparameters,
            train_params: TrainParameters,
    ):
        self.hyperparams = hyperparams
        self.train_param = train_params
        self.model = GPTLanguageModel(self.hyperparams).to(device)
        self.m = model.to(device)

    def Generate_Text(
            self,
            context: str,
            max_new_tokens: int = 500,
    ):
        generated_text = ""
        context = torch.tensor(self.hyperparams.encoder(context), device=device)[
                  None, :
                  ]
        generated_text += self.hyperparams.decoder(
            self.m.generate(context, max_new_tokens)[0].tolist()
        )
        return generated_text

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.train_param.eval_iters)
            for k in range(self.train_param.eval_iters):
                X, Y = get_batch(
                    split,
                    trainParameters=self.train_param,
                    hyperparams=self.hyperparams,
                )
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        model = self.model
        hyperparams = self.hyperparams
        checkpoints = self.train_param.checkpoint
        visualate = self.train_param.visualate
        dirName = f"checkpoints/{time.time()}"
        train_losses = []
        val_losses = []
        iter_values = []
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.train_param.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_param.max_iters
        )
        eval_interval = self.train_param.eval_interval
        max_iters = self.train_param.max_iters

        if not os.path.exists(dirName):
            os.makedirs(dirName)

        hyperparams_text = {
            'vocab_size': hyperparams.vocab_size,
            'n_embd': hyperparams.n_embd,
            'n_head': hyperparams.n_head,
            'n_layer': hyperparams.n_layer,
            'dropout': hyperparams.dropout,
            'batch_size': hyperparams.dropout,
            'block_size': hyperparams.block_size,
            'decoder': hyperparams.decoder,
            'encoder': hyperparams.encoder,
            'device': hyperparams.device,
        }

        # Convert hyperparameters to a list of dictionaries
        data = [hyperparams_text]

        # Define headers
        headers = list(hyperparams_text.keys())
        # Save hyperparameters to CSV
        save_to_csv(headers, data, file_path=f"/checkpoints/{dirName}/HyperParams.csv")

        print(sum(p.numel() for p in self.m.parameters()) / 1e6, "M parameters")
        for iter in range(self.train_param.max_iters):
            start_time = time.time()

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = self.estimate_loss()
                train_loss = losses["train"]
                val_loss = losses["val"]
                print(
                    f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                iter_values.append(iter)
                print(self.Generate_Text("my name is", 5))

                if iter != 0 and checkpoints != 0 and iter % checkpoints == 0:
                    path_name = dirName + f"/checkpoint:{iter}.pkl"
                    torch.save(
                        model.state_dict(),
                        path_name,
                    )
                    print("checkpoint successfuly saved")

                print("------------------------------------")

            # sample a batch of data
            xb, yb = get_batch("train", self.hyperparams, self.train_param)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            end_time = time.time()
            print(f"Geçirilen zaman: {end_time - start_time}")

        # Plot losses
        if visualate:
            plt.plot(iter_values, train_losses, label="Train Loss")
            plt.plot(iter_values, val_losses, label="Val Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title("Train and Validation Loss")
            plt.legend()
            plt.show()
        torch.save(self.model.state_dict(), "model_weights.pth")
        print("Model weights saved successfully")

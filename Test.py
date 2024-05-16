from models.GPTModel import GPTLanguageModel
import torch
import time
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open("input2.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split the text into words

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
vocab_size = vocab_size
n_embd = 256
n_head = 6
n_layer = 20
dropout = 0.3
batch_size = 128
block_size = 128
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
checkpoint = 10
eval_interval = 1
eval_iters = 100
graficate = False
text = text

model = GPTLanguageModel(block_size, n_embd, decoder, n_head, n_layer, vocab_size, dropout).to(device)
m = model.to(device)
print("Device of model:", next(model.parameters()).device)
print("Train parameters device:", device)
print("Hyper parameters device:", device)


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    _data = train_data if split == "train" else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix]).to(device)
    x, y = x.to(device), y.to(device)
    return x, y


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


def train():
    dir_name = f"checkpoints/{time.time()}"
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


train()

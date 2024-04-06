from models.TransformerModel import Transformer  # this is the transformer.py file
import torch
import numpy as np
import matplotlib.pyplot as plt


translate_file = "TR2EN.txt"

turkish_sentences = []
english_sentences = []

with open(translate_file, "r", encoding="utf8") as file:
    lines = file.readlines()
    for line in lines:
        words = line.strip().split("\t")
        if len(words) == 2:
            turkish_sentences.append(words[1])
            english_sentences.append(words[0])


START_TOKEN = "<START>"
PADDING_TOKEN = "<PAD>"
END_TOKEN = "<END>"
turkish_vocabulary = [
    START_TOKEN,
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "ç",
    "d",
    "e",
    "f",
    "g",
    "ğ",
    "h",
    "ı",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ö",
    "p",
    "q",
    "r",
    "s",
    "ş",
    "t",
    "u",
    "ü",
    "v",
    "w",
    "x",
    "y",
    "z",
    PADDING_TOKEN,
    END_TOKEN,
]

english_vocabulary = [
    START_TOKEN,
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
    PADDING_TOKEN,
    END_TOKEN,
]

index_to_turkish = {k: v for k, v in enumerate(turkish_vocabulary)}
turkish_to_index = {v: k for k, v in enumerate(turkish_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}


# Limit Number of sentences
TOTAL_SENTENCES = 20000000
english_sentences = english_sentences[:TOTAL_SENTENCES]
turkish_sentences = turkish_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip("\n").lower() for sentence in english_sentences]
turkish_sentences = [sentence.rstrip("\n") for sentence in turkish_sentences]


import numpy as np

PERCENTILE = 97
print(
    f"{PERCENTILE}th percentile length Turkish: {np.percentile([len(x) for x in turkish_sentences], PERCENTILE)}"
)
print(
    f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}"
)


max_sequence_length = 200


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (
        max_sequence_length - 1
    )  # need to re-add the end token so leaving 1 space


valid_sentence_indicies = []
for index in range(len(turkish_sentences)):
    turkish_sentence, english_sentence = (
        turkish_sentences[index],
        english_sentences[index],
    )
    if (
        is_valid_length(turkish_sentence, max_sequence_length)
        and is_valid_length(english_sentence, max_sequence_length)
        and is_valid_tokens(turkish_sentence, turkish_vocabulary)
    ):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(turkish_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

turkish_sentences = [turkish_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]


import torch

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
kn_vocab_size = len(turkish_vocabulary)

transformer = Transformer(
    d_model,
    ffn_hidden,
    num_heads,
    drop_prob,
    num_layers,
    max_sequence_length,
    kn_vocab_size,
    english_to_index,
    turkish_to_index,
    START_TOKEN,
    END_TOKEN,
    PADDING_TOKEN,
)

from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):

    def __init__(self, english_sentences, turkish_sentences):
        self.english_sentences = english_sentences
        self.turkish_sentences = turkish_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.turkish_sentences[idx]


dataset = TextDataset(english_sentences, turkish_sentences)

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)


for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break


from torch import nn

criterian = nn.CrossEntropyLoss(
    ignore_index=turkish_to_index[PADDING_TOKEN], reduction="none"
)

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


NEG_INFTY = -1e9


def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False
    )
    decoder_padding_mask_self_attention = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False
    )
    decoder_padding_mask_cross_attention = torch.full(
        [num_sentences, max_sequence_length, max_sequence_length], False
    )

    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(
            kn_batch[idx]
        )
        eng_chars_to_padding_mask = np.arange(
            eng_sentence_length + 1, max_sequence_length
        )
        kn_chars_to_padding_mask = np.arange(
            kn_sentence_length + 1, max_sequence_length
        )
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(
        look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0
    )
    decoder_cross_attention_mask = torch.where(
        decoder_padding_mask_cross_attention, NEG_INFTY, 0
    )
    return (
        encoder_self_attention_mask,
        decoder_self_attention_mask,
        decoder_cross_attention_mask,
    )


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 100

train_losses = []


def train():

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        iterator = iter(train_loader)
        epoch_loss = 0.0  # Her epoch için toplam loss değerini saklamak için
        for batch_num, batch in enumerate(iterator):
            transformer.train()
            eng_batch, kn_batch = batch
            (
                encoder_self_attention_mask,
                decoder_self_attention_mask,
                decoder_cross_attention_mask,
            ) = create_masks(eng_batch, kn_batch)
            optim.zero_grad()
            kn_predictions = transformer(
                eng_batch,
                kn_batch,
                encoder_self_attention_mask.to(device),
                decoder_self_attention_mask.to(device),
                decoder_cross_attention_mask.to(device),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True,
            )
            labels = transformer.decoder.sentence_embedding.batch_tokenize(
                kn_batch, start_token=False, end_token=True
            )
            loss = criterian(
                kn_predictions.view(-1, kn_vocab_size).to(device),
                labels.view(-1).to(device),
            ).to(device)
            valid_indicies = torch.where(
                labels.view(-1) == turkish_to_index[PADDING_TOKEN], False, True
            )
            loss = loss.sum() / valid_indicies.sum()
            loss.backward()
            optim.step()
            epoch_loss += (
                loss.item()
            )  # Her batch için loss değerini toplam epoch lossuna ekle
            if batch_num % 100 == 0:
                print(f"Iteration {batch_num} : {loss.item()}")
            # train_losses.append(loss.item())
            if batch_num % 100 == 0:
                print(f"Iteration {batch_num} : {loss.item()}")
                print(f"English: {eng_batch[0]}")
                print(f"Turkish Translation: {kn_batch[0]}")
                kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in kn_sentence_predicted:
                    if idx == turkish_to_index[END_TOKEN]:
                        break
                    predicted_sentence += index_to_turkish[idx.item()]
                print(f"Turkish Prediction: {predicted_sentence}")

                transformer.eval()
                kn_sentence = ("",)
                eng_sentence = ("should we go to the mall?",)
                for word_counter in range(max_sequence_length):
                    (
                        encoder_self_attention_mask,
                        decoder_self_attention_mask,
                        decoder_cross_attention_mask,
                    ) = create_masks(eng_sentence, kn_sentence)
                    predictions = transformer(
                        eng_sentence,
                        kn_sentence,
                        encoder_self_attention_mask.to(device),
                        decoder_self_attention_mask.to(device),
                        decoder_cross_attention_mask.to(device),
                        enc_start_token=False,
                        enc_end_token=False,
                        dec_start_token=True,
                        dec_end_token=False,
                    )
                    next_token_prob_distribution = predictions[0][
                        word_counter
                    ]  # not actual probs
                    next_token_index = torch.argmax(next_token_prob_distribution).item()
                    next_token = index_to_turkish[next_token_index]
                    kn_sentence = (kn_sentence[0] + next_token,)
                    if next_token == END_TOKEN:
                        break

                print(
                    f"Evaluation translation (should we go to the mall?) : {kn_sentence}"
                )
                print("-------------------------------------------")
                train_losses.append(epoch_loss / len(train_loader))
                torch.save(transformer.state_dict(), "model_weights.pkl")

    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()


#train()


transformer.eval()


def translate(eng_sentence):
    eng_sentence = (eng_sentence,)
    kn_sentence = ("",)
    for word_counter in range(max_sequence_length):
        (
            encoder_self_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        ) = create_masks(eng_sentence, kn_sentence)
        predictions = transformer(
            eng_sentence,
            kn_sentence,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=False,
        )
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_turkish[next_token_index]
        kn_sentence = (kn_sentence[0] + next_token,)
        if next_token == END_TOKEN:
            break
    return kn_sentence[0]

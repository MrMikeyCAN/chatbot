from models.TransformerModel import Transformer  # this is the transformer.py file
import torch
import numpy as np


english_file = "test_english.txt"  # replace this path with appropriate one
kannada_file = "test_turkish.txt"  # replace this path with appropriate one

# Generated this by filtering Appendix code

START_TOKEN = "<START>"
PADDING_TOKEN = "<PAD>"
END_TOKEN = "<END>"

kannada_vocabulary = [
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
    "'",
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
    "{",
    "|",
    "}",
    "~",
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
    "'",
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


index_to_kannada = {k: v for k, v in enumerate(kannada_vocabulary)}
kannada_to_index = {v: k for k, v in enumerate(kannada_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}


with open(english_file, "r") as file:
    english_sentences = file.readlines()
with open(kannada_file, "r") as file:
    kannada_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 30
english_sentences = english_sentences[:TOTAL_SENTENCES]
kannada_sentences = kannada_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip("\n").lower() for sentence in english_sentences]
kannada_sentences = [sentence.rstrip("\n") for sentence in kannada_sentences]

print(english_sentences, kannada_sentences)

import numpy as np

PERCENTILE = 97
print(
    f"{PERCENTILE}th percentile length Kannada: {np.percentile([len(x) for x in kannada_sentences], PERCENTILE)}"
)
print(
    f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}"
)


max_sequence_length = 10


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


for index in range(len(kannada_sentences)):
    kannada_sentence, english_sentence = (
        kannada_sentences[index],
        english_sentences[index],
    )
    if (
        is_valid_length(kannada_sentence, max_sequence_length)
        and is_valid_length(english_sentence, max_sequence_length)
        and is_valid_tokens(kannada_sentence, kannada_vocabulary)
    ):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(kannada_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


kannada_sentences = [kannada_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]


kannada_sentences[:3]


import torch

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
kn_vocab_size = len(kannada_vocabulary)

transformer = Transformer(
    d_model,
    ffn_hidden,
    num_heads,
    drop_prob,
    num_layers,
    max_sequence_length,
    kn_vocab_size,
    english_to_index,
    kannada_to_index,
    START_TOKEN,
    END_TOKEN,
    PADDING_TOKEN,
)

from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):

    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.kannada_sentences[idx]


dataset = TextDataset(english_sentences, kannada_sentences)

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)


for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break


from torch import nn

criterian = nn.CrossEntropyLoss(
    ignore_index=kannada_to_index[PADDING_TOKEN], reduction="none"
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
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
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
            labels.view(-1) == kannada_to_index[PADDING_TOKEN], False, True
        )
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        # train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Kannada Translation: {kn_batch[0]}")
            kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in kn_sentence_predicted:
                if idx == kannada_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_kannada[idx.item()]
            print(f"Kannada Prediction: {predicted_sentence}")

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
                next_token = index_to_kannada[next_token_index]
                kn_sentence = (kn_sentence[0] + next_token,)
                if next_token == END_TOKEN:
                    break

            print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
            print("-------------------------------------------")

from models.TransformerModel import Transformer
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

translate_file = "TR2EN.txt"

turkish_sentences = []
english_sentences = []

with open(translate_file, "r") as file:
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

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
kn_vocab_size = len(turkish_vocabulary)

index_to_turkish = {k: v for k, v in enumerate(turkish_vocabulary)}
turkish_to_index = {v: k for k, v in enumerate(turkish_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

model = Transformer(
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

# Modeli oluştururken aynı parametreleri kullanmalısınız
model.load_state_dict(torch.load("model_weights.pkl"))

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


def translate(eng_sentence):
    eng_sentence = (eng_sentence,)
    kn_sentence = ("",)
    for word_counter in range(max_sequence_length):
        (
            encoder_self_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        ) = create_masks(eng_sentence, kn_sentence)
        predictions = model(
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


translation = translate("hi i am mert")
print(translation)

translation = translate("hello how are you")
print(translation)

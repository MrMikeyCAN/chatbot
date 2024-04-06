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


turkish_sentences = []
english_sentences = []

with open("TR2EN.txt", "r", encoding="utf8") as file:
    lines = file.readlines()
    for line in lines:
        words = line.strip().split("\t")
        if len(words) == 2:
            turkish_sentences.append(words[1].lower())
            english_sentences.append(words[0].lower())


def istenmeyen_harfleri_kaldir(metinler, istenmeyen_harfler):
    unwanted = set()  # İstenmeyen harfleri saklamak için bir küme oluşturuyoruz
    for metin in metinler:
        for harf in metin:
            if harf.lower() not in istenmeyen_harfler:
                unwanted.add(
                    harf
                )  # Eğer harf istenmeyen harfler içinde değilse, kümeye ekliyoruz
    return list(unwanted)  # Sonuç olarak bir listeyi döndürüyoruz


# Kullanım örneği:
temiz_metin = istenmeyen_harfleri_kaldir(english_sentences, english_vocabulary)
print(temiz_metin)

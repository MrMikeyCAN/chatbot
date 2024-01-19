import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

with open("text.txt", "r", encoding="utf-8") as myfile:
    mytext = myfile.read()


mytokenizer = Tokenizer()
mytokenizer.fit_on_texts([mytext])
total_words = len(mytokenizer.word_index) + 1
mytokenizer.word_index


my_input_sequences = []
for line in mytext.split("\n"):
    # print(line)
    token_list = mytokenizer.texts_to_sequences([line])[0]
    # print(token_list)
    for i in range(1, len(token_list)):
        my_n_gram_sequence = token_list[: i + 1]
        # print(my_n_gram_sequence)
        my_input_sequences.append(my_n_gram_sequence)
        # print(input_sequences)
max_sequence_len = max([len(seq) for seq in my_input_sequences])
input_sequences = np.array(
    pad_sequences(my_input_sequences, maxlen=max_sequence_len, padding="pre")
)


X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation="softmax"))
print(model.summary())


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=1)


def nwp(input_text: str, predict_next_words: int) -> str:
    for _ in range(predict_next_words):
        token_list = mytokenizer.texts_to_sequences([input_text])[0]
        print(token_list)
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in mytokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word
        return input_text


print(nwp("Hello sir", 10))

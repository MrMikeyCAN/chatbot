
import numpy as np
import tensorflow as tf
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

n_mels = 128
input_shape = (None, n_mels)
padded_shape_x, padded_shape_y = [None, n_mels], [None]
num_classes = len(alphabet)+1
batch_size = 32


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.map(lambda x, y: tf.py_function(data_process, [x, y], [tf.float32, tf.int32]),
                                num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.padded_batch(
        batch_size,
        padded_shapes=(padded_shape_x, padded_shape_y),
        padding_values=(0.0, 0),
        drop_remainder=True
    )

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
dev_dataset = dev_dataset.map(lambda x, y: tf.py_function(data_process, [x, y], [tf.float32, tf.int32]),
                                num_parallel_calls=tf.data.AUTOTUNE)
dev_dataset = dev_dataset.padded_batch(
        batch_size,
        padded_shapes=(padded_shape_x, padded_shape_y),
        padding_values=(0.0, 0),
        drop_remainder=True
    )

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(shape=input_shape))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='gelu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0))
model.add(tf.keras.layers.Dense(128, activation='gelu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0))
model.add(tf.keras.layers.Dense(128, activation='gelu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0))
model.add(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0))
model.add(tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.01)))

print(model.summary())

model.compile(loss="ctc", optimizer="adam")

model.fit(train_dataset, validation_data=dev_dataset, epochs=2)

for x, y in dev_dataset.take(10):
    predict = model.predict(x)
    predict = tf.transpose(predict, [1, 0, 2])
    predict = tf.keras.activations.softmax(predict, axis=-1)
    sequence_length = tf.constant([predict.shape[0]]*predict.shape[1], dtype=tf.int32)
    decode, log_probs = tf.nn.ctc_beam_search_decoder(predict, sequence_length)
    pred = ""
    for seq in decode:
        for c in seq.indices:
            print(c[0].numpy(), c[1].numpy())
    break



"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
"""
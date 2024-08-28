
import numpy as np
import tensorflow as tf
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



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
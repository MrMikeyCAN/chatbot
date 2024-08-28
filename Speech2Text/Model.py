import tensorflow as tf
from tensorflow.keras import layers, models
from Data_Process import data_processor, train_dataset, dev_dataset
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.1),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.1),
        layers.LSTM(units=64, return_sequences=True, use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                    recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                    dropout=0.1,
                    recurrent_dropout=0.1),
        layers.BatchNormalization(),
        layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.1),
        layers.Dense(units=num_classes, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    ])
    return model


# Create and compile the model
model = create_model(data_processor.input_shape, data_processor.num_classes)
model.compile(optimizer="adam",loss="ctc")

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=20,  # Adjust as needed
    validation_data=dev_dataset,
    callbacks=callbacks
)

# Save the final model
model.save('final_speech_recognition_model.keras')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['lr'], label='Learning Rate')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()

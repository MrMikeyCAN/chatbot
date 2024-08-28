import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Tuple, List


class DataProcessor:
    def __init__(self, **kwargs):
        self.datasets = {}

        # Default parameters optimized for embedded systems with NVIDIA GPUs
        default_params = {
            'audio_feature': 'mel',
            'target_sample_rate': 16000,  # Increased for better quality, still manageable for embedded systems
            'n_mfcc': 20,
            'n_fft': 512,  # Increased for better frequency resolution
            'hop_length': 160,  # Adjusted for overlap
            'win_length': 400,  # Adjusted window length
            'n_mels': 64,  # Increased for better frequency resolution
            'fmin': 20,
            'fmax': 7000,  # Increased upper frequency limit
            'power': 2.0,
            'ref': np.max,
            'top_db': 80.0,
            'max_audio_length': 15,  # Increased maximum audio length (seconds)
            'pre_emphasis': 0.97,
            'n_chroma': 12,
            'n_spectral_centroid': 1,
            'n_spectral_rolloff': 1,
            'batch_size': 64  # Increased batch size for GPU processing
        }

        self.params = {**default_params, **kwargs}
        self.input_shape = None
        self.num_classes = None

    def add_dataset(self, csv_file: str, name: str = None, sep: str = None) -> None:
        """Add a dataset. Uses CSV filename if no name is specified."""
        try:
            self.datasets[name if name is not None else os.path.splitext(csv_file)[0]] = pd.read_csv(csv_file, sep=sep)
            print(f"Dataset '{name}' added successfully.")
        except FileNotFoundError:
            print(f"Error: File '{csv_file}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: File '{csv_file}' is empty.")
        except Exception as e:
            print(f"Error adding dataset: {str(e)}")

    def get_sample(self, name: str, frac: float = 1, reset: bool = False) -> None:
        """Sample and shuffle the dataset. Default is to use the entire dataset."""
        if name not in self.datasets:
            print(f"Error: Dataset '{name}' not found.")
            return

        if not 0 <= frac <= 1:
            print("Error: 'frac' value must be between 0 and 1.")
            return

        try:
            self.datasets[name] = self.datasets[name].sample(frac=frac).reset_index(drop=reset)
            print(f"Dataset '{name}' sampled successfully.")
        except Exception as e:
            print(f"Error sampling dataset: {str(e)}")

    def get_features(self, name: str, feature: str) -> pd.Series:
        """Return the requested feature from the dataset."""
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found.")

        if feature not in self.datasets[name].columns:
            raise KeyError(f"Feature '{feature}' not found in dataset '{name}'.")

        return self.datasets[name][feature]

    def set_alphabet(self, alphabet: str = None, file_name: str = None) -> None:
        """Set the alphabet. Either directly or from a file."""
        if alphabet is not None:
            self.alphabet = alphabet.strip()
        elif file_name is not None:
            try:
                with open(file_name, 'r', encoding='utf8') as alphabet_file:
                    self.alphabet = ''.join(alphabet_file.readlines())
            except FileNotFoundError:
                raise FileNotFoundError(f"Alphabet file '{file_name}' not found.")
        else:
            raise ValueError("Either 'alphabet' or 'file_name' must be provided.")

        if len(self.alphabet) != 95:
            raise ValueError("Invalid alphabet length. Expected 95 characters.")

        self.num_classes = len(self.alphabet) + 1  # +1 for blank token in CTC loss
        print(f"Alphabet set successfully. Number of classes: {self.num_classes}")

    def audio_split(self, y: np.array, sr: int) -> List[np.array]:
        """Split audio into segments of maximum duration."""
        max_length = int(self.params['max_audio_length'] * sr)
        return [y[i:i + max_length] for i in range(0, len(y), max_length)]

    def text_split(self, text: str, segments: List[np.array], audio_len: int) -> List[str]:
        """Split text according to audio segments."""
        total_length = len(text)
        segment_ratios = [len(seg) / audio_len for seg in segments]

        text_segments = []
        start = 0
        for ratio in segment_ratios:
            end = min(start + int(ratio * total_length), total_length)
            text_segments.append(text[start:end])
            start = end

        return text_segments

    def extract_audio_features(self, y: np.array, sr: int) -> np.array:
        """Extract audio features based on the selected feature type."""
        y_emphasized = librosa.effects.preemphasis(y, coef=self.params['pre_emphasis'])

        feature_extractors = {
            "mfcc": lambda: librosa.feature.mfcc(
                y=y_emphasized, sr=sr, n_mfcc=self.params['n_mfcc'],
                n_fft=self.params['n_fft'], hop_length=self.params['hop_length'],
                win_length=self.params['win_length'], n_mels=self.params['n_mels']
            ).T,
            "mel": lambda: librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y_emphasized, sr=sr, n_fft=self.params['n_fft'],
                    hop_length=self.params['hop_length'], win_length=self.params['win_length'],
                    n_mels=self.params['n_mels'], fmin=self.params['fmin'], fmax=self.params['fmax'],
                    power=self.params['power']
                ),
                ref=self.params['ref'], top_db=self.params['top_db']
            ).T,
            "chroma": lambda: librosa.feature.chroma_stft(
                y=y_emphasized, sr=sr, n_chroma=self.params['n_chroma'],
                n_fft=self.params['n_fft'], hop_length=self.params['hop_length']
            ).T,
            "spectral_centroid": lambda: librosa.feature.spectral_centroid(
                y=y_emphasized, sr=sr, n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length']
            ).T,
            "spectral_rolloff": lambda: librosa.feature.spectral_rolloff(
                y=y_emphasized, sr=sr, n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length']
            ).T,
            "zero_crossing_rate": lambda: librosa.feature.zero_crossing_rate(
                y=y_emphasized, frame_length=self.params['n_fft'],
                hop_length=self.params['hop_length']
            ).T
        }

        if self.params["audio_feature"] not in feature_extractors:
            raise KeyError(f"Selected feature '{self.params['audio_feature']}' not found.")

        features = feature_extractors[self.params["audio_feature"]]()

        if self.input_shape is None:
            self.input_shape = (None, features.shape[1])
            print(f"Input shape set to: {self.input_shape}")

        return features

    def data_process(self, x_data: str, y_data: str) -> Tuple[List[np.array], List[np.array]]:
        """Process audio and text data."""
        try:
            audio, sr = librosa.load(x_data, sr=self.params['target_sample_rate'])
            audio = librosa.util.normalize(audio)

            audio_segments = self.audio_split(audio, sr)
            text_segments = self.text_split(y_data, audio_segments, len(audio))

            processed_audio = [self.extract_audio_features(seg, sr) for seg in audio_segments]
            processed_labels = [np.array([self.alphabet.index(c) + 1 for c in seg]) for seg in text_segments]

            return processed_audio, processed_labels
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return [], []

    def prepare_dataset(self, name: str, x_feature: str, y_feature: str, dataset_path: str) -> Tuple[List, List]:
        """Prepare dataset for training."""
        try:
            data_x = self.get_features(name, feature=x_feature)
            data_y = self.get_features(name, feature=y_feature)

            datasets_x = []
            datasets_y = []

            for x, y in zip(data_x, data_y):
                x_output, y_output = self.data_process(os.path.join(dataset_path, x), y)
                datasets_x.extend(x_output)
                datasets_y.extend(y_output)

            return datasets_x, datasets_y
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            return [], []

    def get_dataset(self, data_x: List, data_y: List, x_padding_value: float = -80, y_padding_value: int = 0) -> tf.data.Dataset:
        """Create a TensorFlow dataset with dynamic padding."""

        def generator():
            for x, y in zip(data_x, data_y):
                yield x, y

        output_signature = (
            tf.TensorSpec(shape=(None, self.input_shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        padded_shapes = ([None, self.input_shape[1]], [None])
        padding_values = (tf.constant(x_padding_value, dtype=tf.float32), tf.constant(y_padding_value, dtype=tf.int32))

        dataset = dataset.padded_batch(
            self.params['batch_size'],
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True
        )

        return dataset.prefetch(tf.data.AUTOTUNE)


"""
# Usage example
data_path = "cv-corpus-18.0-2024-06-14/tr"
data_processor = DataProcessor()

data_processor.add_dataset(os.path.join(data_path, "train.tsv"), name='train', sep='\t')
data_processor.get_sample('train', frac=0.01, reset=True)
data_processor.set_alphabet(file_name="alphabet.txt")

train_x, train_y = data_processor.prepare_dataset("train", x_feature="path", y_feature="sentence",
                                                  dataset_path=os.path.join(data_path, "clips"))
train_dataset = data_processor.get_dataset(train_x, train_y, x_padding_value=-80, y_padding_value=0)

print(f"Input shape: {data_processor.input_shape}")
print(f"Number of classes: {data_processor.num_classes}")

for x, y in train_dataset:
    print(f"Batch shapes - X: {x.shape}, Y: {y.shape}")
"""

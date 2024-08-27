import librosa
import numpy as np
import pandas as pd
import os


class DataProcessor:
    def __init__(self):
        self.datasets = {}

        self.default_params = {
            'audio_feature': 'mel',
            'target_sample_rate': 16000,
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 128,
            'fmin': 0.0,
            'fmax': None,
            'power': 2.0,
            'ref': np.max,
            'top_db': 80.0
        }

    def add_dataset(self, csv_file:str ,name:str=None, sep:str=None)->None:
        """
        Burası yeni özelik atamamızı sağlar isim isteğe bağlı eklenmez ise csv_file name alır
        """
        self.datasets[name if name is not None else csv_file[:-4]] = pd.read_csv(csv_file, sep=sep)

    def get_sample(self, name:str, frac:float, reset:bool=False)->None:
        """
        Burası örnek almak ve karıştırmak için çağrılmaz ise tümü alınır
        """
        assert 0 <= frac <= 1, "frac değeri 0 ile 1 arasında olmalıdır."
        self.datasets[name].sample(frac=frac).reset_index(drop=reset)

    def get_features(self, name:str, feature:str)->pd.DataFrame:
        """
        Bu kod veri setlerinden istenen özelliği döndürür
        """
        return self.datasets[name][feature].values

    def __read_alphabet(self, file_name:str):
        """
        Alfabe oluştururken dosyadan okumak için eklendi private dosyadır
        """
        try:
            with open(file_name, 'r', encoding='utf8') as alphabet_file:
                self.alphabet = ''.join(alphabet_file.readlines())
                if len(alphabet) != 95:
                    raise ValueError('Alfabe yok')
        except FileNotFoundError:
            raise FileNotFoundError('Alfabe dosyası yok')

    def set_alphabet(self, alphabet:str=None, file_name:str=None)->None:
        """
        Burada alfebeye oluşturmak için eklendi ya alfabe eklenir yada dosyadan okunur
        """
        if alphabet is not None:
            self.alphabet = alphabet
        elif file_name is not None:
            self.__read_alphabet(file_name)
        else:
            raise ValueError('Alfabe oluşturulamadı')

    def audio_split(self, y: np.array, sr: int, max_sec: int)->list:
        """
        Sesi belirlenen süreye göre parçalamak burada önce uzunluk kontorlü yapar eğer kısa ise olduğu gibi alır
        uzun ise segment kadar ilerleyerek başlar sonra ilerleyebileceği uzunluk biterse geriye kalan hepsini alır
        NOT: Bellek yönetimi için çok önemli
        """
        segment_length = int(max_sec * sr)

        total_length = len(y)

        if total_length <= segment_length:
            return [y]

        segments = []
        for start in range(0, total_length, segment_length):
            end = start + segment_length

            if end > total_length:
                segment = y[start:]
            else:
                segment = y[start:end]

            segments.append(segment)

        return segments

    def text_split(self, y: np.array, sr: int, max_sec: int) -> list:
        """
        Sesi belirlenen süreye göre parçalamak burada önce uzunluk kontorlü yapar eğer kısa ise olduğu gibi alır
        uzun ise segment kadar ilerleyerek başlar sonra ilerleyebileceği uzunluk biterse geriye kalan hepsini alır
        NOT: Bellek yönetimi için çok önemli
        """
        segment_length = int(max_sec * sr)

        total_length = len(y)

        if total_length <= segment_length:
            return [y]

        segments = []
        for start in range(0, total_length, segment_length):
            end = start + segment_length

            if end > total_length:
                segment = y[start:]
            else:
                segment = y[start:end]

            segments.append(segment)

        return segments
    def data_process(self, x_data, y_data, **kwargs):

        params = {**self.default_params, **kwargs}

        # Audio
        audio, sr = librosa.load(x_data, sr=params['target_sample_rate'])
        audio = librosa.util.normalize(audio)

        """if params["audio_feature"] == "mfcc":
            audio_array = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=params['n_mfcc'],
                n_fft=params['n_fft'],
                hop_length=params['hop_length'],
                n_mels=params['n_mels']
            )
            audio_array = audio_array.T
        elif params["audio_feature"] == "mel":
            audio_array = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=params['n_fft'],
                hop_length=params['hop_length'],
                n_mels=params['n_mels'],
                fmin=params['fmin'],
                fmax=params['fmax'],
                power=params['power']
            )
            audio_array = librosa.power_to_db(audio_array, ref=params['ref'], top_db=params['top_db']).T
        """
        # Label
        y_data = tf.compat.as_str(y_data.numpy())
        label_array = np.array([alphabet.index(c) + 1 for c in y_data])

        return audio_array, label_array

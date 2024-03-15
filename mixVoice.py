import os
import random
from pydub import AudioSegment

# Kaynak klasörler
klasor1 = "voices1"
klasor2 = "voices2"

# Çıkış klasörü
cikis_klasoru = "mixed"

# Çıkış klasörünün var olup olmadığını kontrol et, yoksa oluştur
if not os.path.exists(cikis_klasoru):
    os.makedirs(cikis_klasoru)

# Klasörlerdeki dosyaları listele
dosyalar1 = os.listdir(klasor1)
dosyalar2 = os.listdir(klasor2)

# Her bir dosya için klasör1'den bir dosya al ve klasör2'den rastgele bir dosya ile birleştir
for dosya1 in dosyalar1:
    # İlk ses dosyasını yükle
    ses1 = AudioSegment.from_file(os.path.join(klasor1, dosya1))

    # Klasör2'den rastgele bir dosya seç
    rastgele_dosya = random.choice(dosyalar2)

    # İkinci ses dosyasını yükle
    ses2 = AudioSegment.from_file(os.path.join(klasor2, rastgele_dosya))

    # İkinci ses dosyasının ses seviyesini ayarla
    ses2_ses_seviyesi_dusurulmus = ses2 - 10  # Örnek: ses seviyesini -10 dB düşür

    # İki ses dosyasını üst üste ekle
    birlesik_ses = ses1.overlay(ses2_ses_seviyesi_dusurulmus)

    # Birleştirilmiş ses dosyasını kaydet
    cikis_dosyasi_adi = f"birlesik_{dosya1}"
    birlesik_ses.export(os.path.join(cikis_klasoru, cikis_dosyasi_adi), format="mp3")

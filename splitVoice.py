import os
import random
import shutil

# Kaynak klasör ve hedef klasörlerin yollarını belirtin
kaynak_klasor = "tr/clips"
hedef_klasor1 = "voices1"
hedef_klasor2 = "voices2"

# Hedef klasörlerin var olup olmadığını kontrol et, yoksa oluştur
if not os.path.exists(hedef_klasor1):
    os.makedirs(hedef_klasor1)
if not os.path.exists(hedef_klasor2):
    os.makedirs(hedef_klasor2)

# Kaynak klasördeki dosyaları alın
dosyalar = os.listdir(kaynak_klasor)

# Dosyaları rastgele karıştır
random.shuffle(dosyalar)

# Dosyaları belirli bir yüzdeye göre ikiye ayır
yuzde = 0.7  # Örnek olarak %70
ayirma_noktasi = int(len(dosyalar) * yuzde)

# İlk bölümü hedef_klasor1'e, ikinci bölümü hedef_klasor2'ye kopyala
for i, dosya in enumerate(dosyalar):
    dosya_yolu = os.path.join(kaynak_klasor, dosya)
    if i < ayirma_noktasi:
        hedef_klasor = hedef_klasor1
    else:
        hedef_klasor = hedef_klasor2
    shutil.copy(dosya_yolu, hedef_klasor)

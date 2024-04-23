Yapay zekanın eğitim aşamaları

1-Öncelikle Speech2Text içerisinde Datasets ve Datasets içerisinde en az 2 tane dil dosyası olamsı gerek
    yükleme:
        Dil veri setleri common voiceden alınmıştır veri setlernini "https://commonvoice.mozilla.org/" adresinden
        common voice 4 versiyonunu yükleyebilirsiniz
    kurulum:
        Datasets içerisinde dilin kısaltılmış adı ile klasör açılır içerisine indirilen dosya eklenir

        windows için:
            cmd ile dosyanın bulunduğu konuma gidilirek "tar xvzf dosya_adi".tar yazılır
        macOS için:
            terminal ile dosyanın bulunduğu konuma gidilirek "tar xvzf dosya_adi.tar" yazılır
        linux için:
            terminal ile dosyanın bulunduğu konuma gidilirek "tar xvzf dosya_adi".tar yazılır
        
        bu işlem tüm yüklenen veriler için tekrarlanır

2-Veri seti kurulumu bittikten sonra yapılması gereken işlem kütüphanelerin kurulumudur
    rust kurulumu:
        windows için:
            "https://rustup.rs/" adresine gidip exe dosyasını indirip kurulumu tamamlıyoruz
        macOS için:
            terminale "curl –-proto ‘=https’ –tlsv1.2 -sSf https://sh.rustup.rs | sh" komutunu yazıp kurulumu tamamlıyoruz
        Linux için:
            terminale "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" komutunu yazıp kurulumu tamamlıyoruz
    python kütüphane kurulumları:
        terminali açıp Speech2Text konumuna gidiyoruz
        "pip/pip3 install -r requirements.txt" yazıp python kütüphanelerini kuruyoruz
        tensorflowu cuda ile kuracağımız için ayrı kurucağız

        1-"conda install -c conda-forge cudatoolkit=cuda_surumunuz cudnn="varsa_cudnn_surumunuz""

        2-"python -m pip install "tensorflow<2.11""

        3-"python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))""

        burdaki satırları terminalde ayrı ayrı çalıştırarak
        tensorflowu windows ortamında cuda ile kurabilir ve test edebilirsiniz

        NOT: conda kurulu değilse kurulmalı ve cudnn yoksa silinmelidir 
        
        1-"python3 -m pip install tensorflow[and-cuda]"
        
        2-"python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))""
        
        burdaki satırları terminalde ayrı ayrı çalıştırarak
        tensorflowu linux ortamında cuda ile kurabilir ve test edebilirsiniz
        
        NOT: cpu ile kurulucaksa terminale "pip/pip3 install tensorflow" yazılması yeterli


3-Yapay zekanın eğitilmesi için veri hazırlığı gerek data_filtering.py dosyasını terminal ile çalıştırıyoruz

4-Yapay zekanın eğitilmesi için Language_detection.ipynb dosyasını açıp run all yapıp tüm komutların çalıştırıyoruz
yapay zeka eğitilmiş oluyor bu işlem biraz uzun sürebilir

5-Yapay zekayı kullanmak için biraz beklemeniz gerek daha yazılıyor

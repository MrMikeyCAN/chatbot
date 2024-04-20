def remove_whitespace(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.readlines()

    cleaned_lines = []

    for line in data:
        # Her satır için başında ve sonunda boşlukları temizle
        line = line.strip()
        
        # Boşlukları temizlenmiş satırı ekleyin (boş satırları atlayın)
        if line:
            cleaned_lines.append(line)

    # Temizlenmiş metni yeni bir dosyaya yaz
    with open(output_file, 'w') as file:
        file.write('\n'.join(cleaned_lines))

    print(f"Boşluklar başarıyla temizlendi. Sonuçlar '{output_file}' dosyasına kaydedildi.")

# Input ve output dosya isimlerini belirt
input_file = 'input_no_speakers.txt'
output_file = 'output.txt'

# Boşlukları temizle ve yeni dosyaya yaz
remove_whitespace(input_file, output_file)

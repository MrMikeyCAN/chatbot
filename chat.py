from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch

# Model ve tokenizer yükleme
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# LabelEncoder yükleme (Önce etiketlerin nasıl kodlandığını bilmek gerekiyor)
data = pd.read_csv('LD.csv')  # Veri setinin yolu
label_encoder = LabelEncoder()
label_encoder.fit(data['labels'])

# Metni tahmin etme fonksiyonu
def predict_language(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction_index  = torch.argmax(outputs.logits, dim=1)
    predicted_language = label_encoder.inverse_transform([prediction_index])[0]
    return predicted_language

# Örnek kullanım
text = input("Cümle:")
predicted_language = predict_language(text, model, tokenizer)
print(f"Predicted Language Index: {predicted_language}")

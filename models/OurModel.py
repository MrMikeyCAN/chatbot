import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder


# TODO Kendi tokenizer'ımızı oluşturuyoruz
class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        # Metni karakterlerine ayırın ve her bir karakterin ASCII değerini alarak bir liste oluşturun
        token_ids = [ord(char) for char in text]
        # Elde edilen tokenleri bir tensor olarak dönüştürün
        tensor = torch.tensor([token_ids])
        return tensor


# TODO language_detection block oluşturuyoruz ki dil algılama sağlanabilsin
class LanguageClassifier:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = torch.load(
            model_path + "/label_encoder_classes.pt"
        )

    def classify_text(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        return predicted_label


# TODO Training arguments oluşturuyoruz ki kodları güvenliği ve temizliği için
class TrainArgumentsForOurModel:
    def __init__(
        self,
        guessLanguage=False,
        embed_size=512,
        heads=8,
        dropout=0,
        forward_expansion=4,
        src_vocab_size=None,
        trg_vocab_size=None,
        src_pad_idx=0,
        trg_pad_idx=0,
        num_layers=6,
        device="cpu",
        max_length=100,
        model_path="./language_detection_model",
        norm_activite_func="LayerNorm",
    ):
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.norm_activite_func = norm_activite_func
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.num_layers = num_layers
        self.device = device
        self.max_length = max_length
        self.model_path = model_path
        self.guessLanguage = guessLanguage


class AttentionBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionBlock, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Başlıklara göre gömülü boyutlarının uygun olup olmadığını kontrol edin
        assert (
            self.head_dim * heads == embed_size
        ), "Gömülü değer belirtilen başlıklara bölünebilmeli"

        # Anahtar, değer ve sorgu matrislerini oluşturun
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = (
            values.shape[1],
            keys.shape[1],
            query.shape[1],
        )

        # Değerlerin boyutunu alın
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Boyutları yeniden şekillendirin
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Sorgu ve anahtar arasındaki dikkat ağırlıklarını hesaplayın
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Maskeyi uygulayın (varsa)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Dikkat ağırlıklarını softmax ile normalize edin
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Ağırlıklı değerlerle birleştirin
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Çıktıyı tam bağlı katmana gönderin
        out = self.fc_out(out)

        return out


class Tanh(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x**3)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = AttentionBlock(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Dikkat mekanizmasını ve katman normalizasyonunu uygulayın
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # Tüm katmanları geçişli olarak uygulayın
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        device,
        norm_activite_func="LayerNorm",
    ):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = AttentionBlock(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
        norm_activite_func,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(self, arguments: TrainArgumentsForOurModel, item):

        super(Transformer, self).__init__()

        self.languageClassifier = LanguageClassifier(model_path=arguments.model_path)

        if arguments.guessLanguage:
            self.language = self.languageClassifier.classify_text(item)

        self.encoder = Encoder(
            arguments.src_vocab_size,
            arguments.embed_size,
            arguments.num_layers,
            arguments.heads,
            arguments.device,
            arguments.forward_expansion,
            arguments.dropout,
            arguments.max_length,
        )

        self.decoder = Decoder(
            arguments.trg_vocab_size,
            arguments.embed_size,
            arguments.num_layers,
            arguments.heads,
            arguments.forward_expansion,
            arguments.dropout,
            arguments.device,
            arguments.max_length,
            arguments.norm_activite_func,
        )

        self.src_pad_idx = arguments.src_pad_idx
        self.trg_pad_idx = arguments.trg_pad_idx
        self.device = arguments.device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Öncelikle, model için bir tokenizer oluşturalım
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Giriş ve çıkış cümlelerimizi tokenize edelim
    input_sentence = "How are you?"
    target_sentence = "Wie geht es dir?"

    input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)
    target_ids = tokenizer.encode(target_sentence, add_special_tokens=True)

    # Tensorlara dönüştürelim
    input_tensor = torch.tensor([input_ids])
    target_tensor = torch.tensor([target_ids])

    # Eğitim için model parametrelerini belirleyelim
    train_arguments = TrainArgumentsForOurModel(
        embed_size=512,
        heads=8,
        dropout=0.1,
        forward_expansion=4,
        src_vocab_size=input_tensor.max().item()
        + 1,  # +1, çünkü 0'dan başlamak yerine 1'den başlıyoruz
        trg_vocab_size=target_tensor.max().item()
        + 1,  # +1, çünkü 0'dan başlamak yerine 1'den başlıyoruz
        num_layers=6,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_length=100,
        norm_activite_func="layernorm",
        guessLanguage=True,
    )

    # Modeli oluşturalım
    model = Transformer(train_arguments, item="Example")

    # Eğitim için kayıp fonksiyonunu belirleyelim
    criterion = nn.CrossEntropyLoss()

    # Optimizasyon fonksiyonunu ve başlangıç öğrenme oranını belirleyelim
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Öğrenme oranını izlemek için bir liste oluşturalım
    learning_rates = []

    # Eğitim verilerini belirleyelim
    input_data = input_tensor.to(train_arguments.device)
    target_data = target_tensor.to(train_arguments.device)

    losses = []

    # Modeli eğitelim
    model.train()
    for epoch in range(50):  # Örnek olarak 50 epoch için eğitim yapalım
        optimizer.zero_grad()
        output = model(
            input_data, target_data[:, :-1]
        )  # Çıkışın son endeksini atlayalım
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = (
            target_data[:, 1:].contiguous().view(-1)
        )  # Etiketlerin ilk endeksini atlayalım
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Loss değerini kaydedelim
        losses.append(loss.item())

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Tabloyu oluşturalım
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.show()


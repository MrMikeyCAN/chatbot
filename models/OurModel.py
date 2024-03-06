import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


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
class TransformerModelArguments:
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
    def __init__(self, arguments: TransformerModelArguments, item):

        super(Transformer, self).__init__()

        if arguments.guessLanguage:
            self.languageClassifier = LanguageClassifier(
                model_path=arguments.model_path
            )
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


### TODO Daha iyi bir sonuç için eğitim için özelleştirilmiş bölümler


class TrainingArguments:
    def __init__(
        self,
        model: nn.Module = None,
        visualize: bool = False,
        epochs: int = 100,
        input_data=None,
        target_data=None,
        learning_rate=0.0001,
        checkpoints=10,
    ):
        self.model = model
        self.visualize = visualize
        self.epochs = epochs
        self.input_data = input_data
        self.target_data = target_data
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints


class Trainer:
    def __init__(self, args: TrainingArguments):
        self.args = args
        self.model = args.model
        self.visualize = args.visualize
        self.epochs = args.epochs
        self.input_data = args.input_data
        self.target_data = args.target_data
        self.learning_rate = args.learning_rate
        self.checkpoints = args.checkpoints

    def train(self):
        losses = []
        self.model.train()
        # Eğitim için kayıp fonksiyonunu belirleyelim
        criterion = nn.CrossEntropyLoss()

        # Optimizasyon fonksiyonunu belirleyelim
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):  # Örnek olarak 100 epoch için eğitim yapalım
            optimizer.zero_grad()
            output = self.model(
                self.input_data, self.target_data[:, :-1]
            )  # Çıkışın son endeksini atlayalım
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = (
                self.target_data[:, 1:].contiguous().view(-1)
            )  # Etiketlerin ilk endeksini atlayalım
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            # Her 10 epoch'ta bir modeli kaydedelim
            if (epoch + 1) % self.checkpoints == 0:
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "epoch": epoch,
                }
                torch.save(
                    checkpoint, f"logs/checkpoint/model_checkpoint_epoch_{epoch+1}.pth"
                )

        if self.visualize:
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss per Epoch")
            plt.show()
        # Eğitim tamamlandıktan sonra modeli kaydedelim
        torch.save(self.model.state_dict(), "trained_model.pth")
        print("Başarıyla kaydedildi")

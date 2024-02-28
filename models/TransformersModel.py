import torch
import torch.nn as nn


### TODO Decoder ve Transformers classlarında kullanmak için AttentionBlock oluşturuyoruz (Mertin değimiyle en zor tek kısım)
class AttentionBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionBlock, self).__init__()  # Yapıcı yöntem çağrısı eklendi
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # * Gömülü boyutların heads sayısına bölünebilir olduğunu kontrol ediyoruz
        assert (
            self.head_dim * heads == embed_size
        ), "Gömülü değer belirtilen başlıklara bölünebilmeli"

        # * v,q,k,o değerlerini oluşturuyoruz
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        # * Değerlerin boyutlarını alıyoruz
        value_len, key_len, query_len = (
            values.shape[1],
            keys.shape[1],
            query.shape[1],
        )

        # * Değerleri ilgili katmanlardan geçiriyoruz
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # * Başlıklara göre gömülü değerleri ayırıyoruz
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # * Attention hesabı için query*key işlemini yapıyoruz
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # * Mask işlemini gerçekleştiriyoruz
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # * Energy değerlerini softmax ile normalize ediyoruz
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # * Son çıktıyı hesaplıyoruz
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


### * Norm işlemleri için farklı iki yol


# TODO GELU aktivasyon fonksiyonunu kullanalım
class Tanh(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x**3)))


# # TODO Group Normalizasyon katmanını kullanalım
# class GroupNormBlock(nn.Module):
#     def __init__(self, embed_size, num_groups=8):
#         super(GroupNormBlock, self).__init__()
#         if embed_size % num_groups != 0:
#             raise ValueError("embed_size should be divisible by num_groups.")
#         if embed_size < num_groups:
#             raise ValueError("embed_size should be greater than or equal to num_groups.")
#         self.norm = nn.GroupNorm(num_groups, embed_size)

#     def forward(self, x):
#         return self.norm(x)


### TODO Transformer block sadece bağlantıyı es geçmek için ve normalize işlemlerini gerçekleştirmek için oluşturduğumuz blok parçası
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        norm_activite_func="LayerNorm",
    ):
        super(TransformerBlock, self).__init__()
        self.attention = AttentionBlock(embed_size, heads)
        if norm_activite_func.lower() == "layernorm":
            self.norm1 = nn.LayerNorm(embed_size)
            self.norm2 = nn.LayerNorm(embed_size)
        elif norm_activite_func.lower() == "tanh":  # elif kullanılmalı
            self.norm1 = Tanh()
            self.norm2 = Tanh()
        # elif norm_activite_func == "Group":  # elif kullanılmalı
        #     self.norm1 = GroupNormBlock(embed_size)
        #     self.norm2 = GroupNormBlock(embed_size)
        else:
            raise ValueError("Geçersiz norm_activite_func değeri")

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # TODO Bağlantıyı es geçmek ve direkt norm işlemlerine uğramak için dropout ekliyoruz
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

        # ! Encoder içerisinde Key Value ve Query AYNI olacaktır ,Decoder içerisinde değişikliğe uğrayacaktır
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
            norm_activite_func=norm_activite_func,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


### * Bu 3 block tamamlandıktan sonra geriye sadece onları decoder ve transformers class içerisinde birleştirmek kalıyor


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
                    norm_activite_func,
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
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        # * Hata vermemesi için belirlenmese bile cpu'da çalışır halde düzenlendi
        device="cpu",
        max_length=100,
        # * Varsayılan değer atıyoruz
        norm_activite_func="LayerNorm",
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
            norm_activite_func,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        device=device,
        norm_activite_func="LayerNorm",
    ).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)

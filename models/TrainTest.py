# from models.OptimizedTransformerModel import Transformer
# import torch
# import numpy as np

# ! Derslerdeki adam kannadaca ile ingilizce arasında translate yaptığından ben de örnekleri onun üzerinden yaptım

# TODO öncelikli olarak bir veri seti bulmalı ve içeriğindeki her biri için index_to ve to_index oluşturulmalı


# TODO sonrasında bir sınır belirlenmeli ve kelimeler küçük harflere çevirilip tokenize edilmeli


# TODO Sonrasında yüzdelik olarak kaç kelimenin belirlenen uzunluğu geçtiği bulunmalı

# * Örnek kullanım
# * import numpy as np
# * PERCENTILE = 97
# * print( f"{PERCENTILE}th percentile length Kannada: {np.percentile([len(x) for x in kannada_sentences], PERCENTILE)}" )
# * print( f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}" )


# TODO sonrasında kelimeleri eşitlemek için pad işlemleri ve tokenize işlemleri


# TODO sonrasında modeli oluşturmalıyız


# TODO sonrasında pytorch'un bize yapmamızı zorladığı dataset kodunu olutşurmalıyız
# * örnek
# * from torch.utils.data import Dataset, DataLoader
# * class TextDataset(Dataset):
# *    def __init__(self, english_sentences, kannada_sentences):
# *        self.english_sentences = english_sentences
# *        self.kannada_sentences = kannada_sentences
# *    def __len__(self):
# *         return len(self.english_sentences)
# *    def __getitem__(self, idx):
# *        return self.english_sentences[idx], self.kannada_sentences[idx]
# * dataset = TextDataset(english_sentences, kannada_sentences)


# TODO sonrasında klasik pytorch dataset uydurmaları
# * örnek
# * train_loader = DataLoader(dataset, batch_size)
# * iterator = iter(train_loader)
# * for batch_num, batch in enumerate(iterator):
# *     print(batch)
# *     if batch_num > 3:
# *         break


# TODO sonrasında loss fonksiyonu optimizasyon fonksiyonu gibi kısımları seçiyoruz


# TODO sonrasında da maskeleme işlemi yapıyoruz
# * örnek
# * NEG_INFTY = -1e9
# * def create_masks(eng_batch, kn_batch):
# *     num_sentences = len(eng_batch)
# *     look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
# *     look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
# *     encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
# *     decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
# *     decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
# *     for idx in range(num_sentences):
# *       eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
# *       eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
# *       kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
# *       encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
# *       encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
# *       decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
# *       decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
# *       decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
# *       decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True
# *     encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
# *     decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
# *     decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
# *     return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

# ! Her ne kadar örnek de desem burada değiştirmemiz gereken tek şey çeviri dili olacak


# TODO şimdi de model ile eğitim yapmalıyız
# ! örnek kodda özellikle nasıl çalıştığını incelememizi için örnek cümlenin her seferinde tekrar tekrar trainler sonucunda çevirisiyle karşılaşıyoruz
# * örnek
# * transformer.train()
# * transformer.to(device)
# * total_loss = 0
# * num_epochs = 10
# * for epoch in range(num_epochs):
# *     print(f"Epoch {epoch}")
# *     iterator = iter(train_loader)
# *     for batch_num, batch in enumerate(iterator):
# *         transformer.train()
# *         eng_batch, kn_batch = batch
# *         encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, kn_batch)
# *         optim.zero_grad()
# *         kn_predictions = transformer(eng_batch,
# *                                      kn_batch,
# *                                      encoder_self_attention_mask.to(device),
# *                                      decoder_self_attention_mask.to(device),
# *                                      decoder_cross_attention_mask.to(device),
# *                                      enc_start_token=False,
# *                                      enc_end_token=False,
# *                                      dec_start_token=True,
# *                                      dec_end_token=True)
# *         labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
# *         loss = criterian(
# *             kn_predictions.view(-1, kn_vocab_size).to(device),
# *             labels.view(-1).to(device)
# *         ).to(device)
# *         valid_indicies = torch.where(labels.view(-1) == kannada_to_index[PADDING_TOKEN], False, True)
# *         loss = loss.sum() / valid_indicies.sum()
# *         loss.backward()
# *         optim.step()
# *         #train_losses.append(loss.item())
# *         if batch_num % 100 == 0:
# *             print(f"Iteration {batch_num} : {loss.item()}")
# *             print(f"English: {eng_batch[0]}")
# *             print(f"Kannada Translation: {kn_batch[0]}")
# *             kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
# *             predicted_sentence = ""
# *             for idx in kn_sentence_predicted:
# *               if idx == kannada_to_index[END_TOKEN]:
# *                 break
# *               predicted_sentence += index_to_kannada[idx.item()]
# *             print(f"Kannada Prediction: {predicted_sentence}")
# *             transformer.eval()
# *             kn_sentence = ("",)
# *             eng_sentence = ("should we go to the mall?",)
# *             for word_counter in range(max_sequence_length):
# *                 encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence)
# *                 predictions = transformer(eng_sentence,
# *                                           kn_sentence,
# *                                           encoder_self_attention_mask.to(device),
# *                                           decoder_self_attention_mask.to(device),
# *                                           decoder_cross_attention_mask.to(device),
# *                                           enc_start_token=False,
# *                                           enc_end_token=False,
# *                                           dec_start_token=True,
# *                                           dec_end_token=False)
# *                 next_token_prob_distribution = predictions[0][word_counter] # not actual probs
# *                 next_token_index = torch.argmax(next_token_prob_distribution).item()
# *                 next_token = index_to_kannada[next_token_index]
# *                 kn_sentence = (kn_sentence[0] + next_token, )
# *                 if next_token == END_TOKEN:
# *                   break
# *             print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
# *             print("-------------------------------------------")


# TODO son olarak da translate için bir fonksiyon oluşturuyoruz
# * örnek
# * transformer.eval()
# * def translate(eng_sentence):
# *   eng_sentence = (eng_sentence,)
# *   kn_sentence = ("",)
# *   for word_counter in range(max_sequence_length):
# *     encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence)
# *     predictions = transformer(eng_sentence,
# *                               kn_sentence,
# *                               encoder_self_attention_mask.to(device),
# *                               decoder_self_attention_mask.to(device),
# *                               decoder_cross_attention_mask.to(device),
# *                               enc_start_token=False,
# *                               enc_end_token=False,
# *                               dec_start_token=True,
# *                               dec_end_token=False)
# *     next_token_prob_distribution = predictions[0][word_counter]
# *     next_token_index = torch.argmax(next_token_prob_distribution).item()
# *     next_token = index_to_kannada[next_token_index]
# *     kn_sentence = (kn_sentence[0] + next_token, )
# *     if next_token == END_TOKEN:
# *       break
# *   return kn_sentence[0]

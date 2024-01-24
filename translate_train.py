from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import evaluate

# Model and Tokenizer
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Dataset
books = load_dataset("opus_books", "tr-en")
books = books["train"].train_test_split(test_size=0.2)

# Preprocessing
prefix = "translate Turkish to English: "

def preprocess_function(examples):
    inputs = [prefix + example["tr"] for example in examples["translation"]]
    targets = [example["en"] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True, text_target=targets)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Metric
metric = evaluate.load("sacrebleu")

# Functions for Postprocessing and Computing Metrics
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_books_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,  # Set to True if you want to push to the Hugging Face Hub
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start Training
trainer.train()

# Save Model and Tokenizer
trainer.save_model("./my_awesome_opus_books_model")
tokenizer.save_pretrained("./my_awesome_opus_books_model")

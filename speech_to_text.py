from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

# Initialize feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Load and preprocess the dataset
minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
minds = minds.train_test_split(test_size=0.2)
minds = minds.remove_columns(
    ["path", "transcription", "english_transcription", "lang_id"]
)

# Map labels to IDs and vice versa
labels = minds["train"].features["intent_class"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# Cast audio column to Audio type
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))


# Preprocessing function for audio
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True,
    )
    return inputs


# Apply preprocessing to the dataset
encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
encoded_minds = encoded_minds.rename_column("intent_class", "label")

# Load accuracy metric
accuracy_metric = evaluate.load("accuracy")


# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(
        predictions=predictions, references=eval_pred.label_ids
    )


# Number of labels
num_labels = len(id2label)

# Initialize the audio classification model
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./my_awesome_audio_classification_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=3000,  # Adjusted for potentially better training
    gradient_checkpointing=True,  # Değiştirildi
    fp16=False,  # Değiştirildi
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,  # Accuracy is higher when it's greater
    push_to_hub=False,  # Set to True if you want to push to the hub
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

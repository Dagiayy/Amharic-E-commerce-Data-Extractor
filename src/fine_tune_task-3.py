# Import libraries
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import evaluate
from collections import defaultdict
from torch import nn
from transformers.trainer_callback import EarlyStoppingCallback
import os
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import default_data_collator

# Step 2: Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use cuda: {device}")

# Step 3: Define label set
label_list = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

# Step 4: Load dataset
def load_conll_data(file_path):
    sentences, labels = [], []
    sentence, sentence_labels = [], []
    line_number = 0
    skipped_lines = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if sentence and sentence_labels:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                    sentence, sentence_labels = [], []
                continue
            try:
                parts = line.split()
                if len(parts) >= 2: # Be more forgiving, take the first two parts if more exist
                    token, label = parts[0], parts[1]
                    if label not in label2id:
                        print(f"Invalid label at line {line_number}: {label}. Skipping line.")
                        skipped_lines.append(line_number)
                        continue
                    sentence.append(token)
                    sentence_labels.append(label)
                else:
                    print(f"Error parsing line {line_number}: Expected at least 2 parts, got {len(parts)}. Skipping line: {line}")
                    skipped_lines.append(line_number)
                    continue
            except Exception as e:
                print(f"Unexpected error processing line {line_number}: {line}. Error: {e}. Skipping line.")
                skipped_lines.append(line_number)
                continue

    if sentence and sentence_labels:
        sentences.append(sentence)
        labels.append(sentence_labels)

    print(f"Loaded {len(sentences)} sentences")
    if skipped_lines:
        print(f"Skipped {len(skipped_lines)} lines due to errors: {skipped_lines[:10]}...") # Print first 10 skipped lines
    if len(sentences) > 0:
        for i, (sent, lbls) in enumerate(zip(sentences[:5], labels[:5])): # Print first 5 sentences
            print(f"Sample Sentence {i+1} length: {len(sent)}, Labels: {len(lbls)}")
    else:
        print("No sentences loaded.")

    label_counts = defaultdict(int)
    total_tokens = 0
    for sentence_labels in labels:
        total_tokens += len(sentence_labels)
        for label in sentence_labels:
            label_counts[label] += 1
    print("Label distribution:", dict(label_counts))
    if total_tokens > 0:
        print("Label percentages:", {k: f"{(v/total_tokens)*100:.2f}%" for k, v in label_counts.items()})
    else:
        print("No tokens processed to calculate percentages.")
    return sentences, labels, label_counts, total_tokens

file_path = "combined_labeled_data.conll"
sentences, labels, label_counts, total_tokens = load_conll_data(file_path)

if len(sentences) == 0:
    raise ValueError("No sentences loaded from the dataset. Please check the file format.")

for i, (sentence, sentence_labels) in enumerate(zip(sentences, labels)):
    assert len(sentence) == len(sentence_labels), f"Mismatch in sentence {i+1}: {sentence}"

labels_ids = [[label2id[label] for label in sentence_labels] for sentence_labels in labels]
data = {"tokens": sentences, "ner_tags": labels_ids}
dataset = Dataset.from_dict(data)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
dataset_dict = DatasetDict({"train": train_test_split["train"], "validation": train_test_split["test"]})
print(f"Train size: {len(dataset_dict['train'])}, Validation size: {len(dataset_dict['validation'])}")

# Step 5: Load tokenizer and model
model_name = "Davlan/afro-xlmr-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if total_tokens > 0 and len(label_counts) > 0:
    class_weights = torch.tensor([total_tokens / (len(label_counts) * label_counts[id2label[i]]) * (10.0 if id2label[i] in ["B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"] else 1.0) for i in range(len(label_list))]).to(device)
    print(f"Class weights: {class_weights}")
else:
    print("Cannot calculate class weights: No tokens processed or label counts are empty.")
    # Handle this case, maybe by using uniform weights or raising an error
    class_weights = torch.ones(len(label_list)).to(device) # Example: Use uniform weights

class WeightedNERModel(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        if labels is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)
            outputs.loss = loss
        return outputs

model = WeightedNERModel.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)
model.gradient_checkpointing_enable()

# Step 6: Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100] * len(word_ids)
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels[idx] = -100
            elif word_idx != prev_word_idx:
                aligned_labels[idx] = label[word_idx]
            else:
                aligned_labels[idx] = -100
            prev_word_idx = word_idx
        labels.append(aligned_labels)
        if i == 0:
            tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][i])
            print(f"Sample tokens: {tokens}")
            print(f"Sample aligned labels: {aligned_labels}")
            entity_tokens = [(tokens[j], id2label[aligned_labels[j]] if aligned_labels[j] != -100 else "IGNORED") for j in range(len(tokens)) if aligned_labels[j] != -100]
            print(f"Tokenized entities: {entity_tokens}")
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

# Step 7: Load evaluation metric
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) and flatten the lists for seqeval
    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    pred_labels = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Ensure corresponding true_labels and pred_labels have the same length
    # This can happen if the -100 filtering results in different lengths
    # We will filter out pairs with unequal lengths
    aligned_true_labels = []
    aligned_pred_labels = []
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        if len(true_seq) == len(pred_seq):
            aligned_true_labels.append(true_seq)
            aligned_pred_labels.append(pred_seq)
        else:
            print(f"Skipping sentence due to length mismatch after filtering: True labels length {len(true_seq)}, Predicted labels length {len(pred_seq)}")


    if not aligned_true_labels:
        print("No aligned labels to compute metrics.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    results = metric.compute(predictions=aligned_pred_labels, references=aligned_true_labels, zero_division=0)

    # Calculate overall accuracy separately, considering ignored tokens
    correct_predictions = 0
    total_predictions = 0
    for prediction, label in zip(predictions, labels):
        for p, l in zip(prediction, label):
            if l != -100:
                total_predictions += 1
                if p == l:
                    correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": accuracy,
    }


# Step 8: Training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=8,
    warmup_steps=5,
    report_to="none",
    gradient_checkpointing=True,
)

# Step 9: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
)

# Step 10: Fine-tune
trainer.train()

# Step 11: Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Step 12: Save model
model.save_pretrained("./ethiomart_ner_model")
tokenizer.save_pretrained("./ethiomart_ner_model")
print("Model saved to ./ethiomart_ner_model")

# Step 13: Inference with post-processing
from transformers import pipeline
ner_pipeline = pipeline(
    "ner",
    model="./ethiomart_ner_model",
    tokenizer="./ethiomart_ner_model",
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)
def filter_predictions(results, min_score=0.5):
    return [pred for pred in results if pred['score'] > min_score]
test_sentences = [
    "ጫማዎች 2800 ብር ቦሌ መድኃኔዓለም",
    "TIMBERLAND 7500 Br አዲስ አበባ",
    "ቡና መፍጫ 2100 ብር መገናኛ",
    "ባህላዊ ልብስ 0974312223 ቦሌ"
]
for sentence in test_sentences:
    results = ner_pipeline(sentence)
    filtered = filter_predictions(results)
    print(f"Inference on '{sentence}':", filtered)

# Debug logits
inputs = tokenizer(test_sentences[0], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predicted_labels = torch.argmax(logits, dim=-1)[0]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Debug logits for first test sentence:")
for token, label_id in zip(tokens, predicted_labels):
    if label_id.item() != -100:
        print(f"Token: {token}, Predicted Label: {id2label[label_id.item()]}")
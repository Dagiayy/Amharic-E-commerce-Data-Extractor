import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import uuid
import time
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device set to use: {device}")

# Load and preprocess dataset
def load_conll_data(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
                continue
            token, label = line.split()
            current_sentence.append(token)
            current_labels.append(label)
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    return sentences, labels

def verify_data(sentences, labels):
    for i in range(min(5, len(sentences))):
        logger.info(f"Sample Sentence {i+1} length: {len(sentences[i])}, Labels: {len(labels[i])}")
    label_counts = defaultdict(int)
    for label_list in labels:
        for label in label_list:
            label_counts[label] += 1
    logger.info(f"Label distribution: {dict(label_counts)}")
    total = sum(label_counts.values())
    label_percentages = {k: f"{(v/total):.2%}" for k, v in label_counts.items()}
    logger.info(f"Label percentages: {label_percentages}")

# Load cleaned dataset
sentences, labels = load_conll_data("cleaned_labeled_data.conll")
verify_data(sentences, labels)

# Label mapping
unique_labels = sorted(set(label for label_list in labels for label in label_list))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

num_labels = len(unique_labels)
logger.info(f"Number of labels: {num_labels}")

# Compute class weights
label_counts = np.array([sum(1 for label_list in labels for label in label_list if label == l) for l in unique_labels])
class_weights = 1.0 / label_counts
class_weights = class_weights * 5.0  # Adjusted multiplier
class_weights = torch.tensor(class_weights, device=device)
logger.info(f"Class weights: {class_weights}")

# Tokenize dataset
def tokenize_and_align_labels(tokenizer, sentences, labels, max_length=512):
    tokenized_inputs = []
    aligned_labels = []
    for sentence, label_list in zip(sentences, labels):
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(" ".join(sentence)) + [tokenizer.sep_token]
        token_labels = [-100] + [label_to_idx[label] for label in label_list] + [-100]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            token_labels = token_labels[:max_length]
        else:
            tokens += [tokenizer.pad_token] * (max_length - len(tokens))
            token_labels += [-100] * (max_length - len(token_labels))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokenized_inputs.append({"input_ids": input_ids, "attention_mask": [1 if token != tokenizer.pad_token else 0 for token in tokens]})
        aligned_labels.append(token_labels)
    logger.info(f"Sample tokens: {tokens[:20]}")
    logger.info(f"Sample aligned labels: {token_labels[:20]}")
    return tokenized_inputs, aligned_labels

# Create dataset
def create_dataset(tokenized_inputs, aligned_labels):
    dataset = Dataset.from_dict({
        "input_ids": [x["input_ids"] for x in tokenized_inputs],
        "attention_mask": [x["attention_mask"] for x in tokenized_inputs],
        "labels": aligned_labels,
    })
    return dataset

# Compute metrics
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])
    precision, recall, f1, _ = precision_recall_f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

# Filter predictions
def filter_predictions(predictions, min_score=0.3):
    entities = []
    for pred in predictions:
        if pred["score"] >= min_score:
            entity = {
                "entity": pred["entity"],
                "score": pred["score"],
                "word": pred["word"],
                "start": pred["start"],
                "end": pred["end"],
            }
            entities.append(entity)
    return entities

# Fine-tune and evaluate model
def fine_tune_model(model_name, output_dir, dataset_train, dataset_val):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    tokenized_inputs, aligned_labels = tokenize_and_align_labels(tokenizer, sentences, labels)
    dataset = create_dataset(tokenized_inputs, aligned_labels)
    train_size = int(0.8 * len(dataset))
    dataset_train = dataset[:train_size]
    dataset_val = dataset[train_size:]
    logger.info(f"Train size: {len(dataset_train)}, Validation size: {len(dataset_val)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
)


    trainer.train()
    eval_results = trainer.evaluate()
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Evaluation Results for {model_name}: {eval_results}")
    logger.info(f"Training time for {model_name}: {training_time:.2f} seconds")

    # Inference speed test
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, aggregation_strategy="first")
    test_sentences = [
        "ጫማዎች 2800 ብር ቦሌ መድኃኔዓለም",
        "TIMBERLAND 7500 ብር አዲስ አበባ",
        "ቡና መፍጫ 2100 ብር መገናኛ",
    ]
    inference_times = []
    results = []
    for sentence in test_sentences:
        start_time = time.time()
        preds = nlp(sentence)
        end_time = time.time()
        inference_times.append(end_time - start_time)
        filtered_preds = filter_predictions(preds)
        results.append(filtered_preds)
        logger.info(f"Inference on '{sentence}' with {model_name}: {filtered_preds}")
    avg_inference_time = np.mean(inference_times)
    logger.info(f"Average inference time for {model_name}: {avg_inference_time:.4f} seconds")

    return {
        "model_name": model_name,
        "eval_results": eval_results,
        "training_time": training_time,
        "avg_inference_time": avg_inference_time,
        "inference_results": results,
    }

# Compare models
models = [
    {"name": "xlm-roberta-base", "output_dir": "./xlm_roberta_ner_model"},
    {"name": "distilbert-base-multilingual-cased", "output_dir": "./distilbert_ner_model"},
    {"name": "bert-base-multilingual-cased", "output_dir": "./mbert_ner_model"},
]

comparison_results = []
for model_info in models:
    result = fine_tune_model(model_info["name"], model_info["output_dir"], None, None)
    comparison_results.append(result)

# Print comparison
logger.info("\nModel Comparison:")
print("| Model | F1 Score | Precision | Recall | Accuracy | Training Time (s) | Inference Time (s) |")
print("|-------|----------|-----------|--------|----------|-------------------|--------------------|")
for result in comparison_results:
    eval_results = result["eval_results"]
    print(
        f"| {result['model_name']} | {eval_results['eval_f1']:.4f} | {eval_results['eval_precision']:.4f} | "
        f"{eval_results['eval_recall']:.4f} | {eval_results['eval_accuracy']:.4f} | {result['training_time']:.2f} | "
        f"{result['avg_inference_time']:.4f} |"
    )

# Select best model
best_model = max(comparison_results, key=lambda x: x["eval_results"]["eval_f1"])
logger.info(f"\nBest Model: {best_model['model_name']}")
logger.info(f"F1 Score: {best_model['eval_results']['eval_f1']:.4f}")
logger.info(f"Training Time: {best_model['training_time']:.2f} seconds")
logger.info(f"Inference Time: {best_model['avg_inference_time']:.4f} seconds")
logger.info(f"Inference Results: {best_model['inference_results']}")
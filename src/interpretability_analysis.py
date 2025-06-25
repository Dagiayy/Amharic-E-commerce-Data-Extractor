import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from shap import Explainer, Explanation
from lime.lime_text import LimeTextExplainer
import logging
from datasets import Dataset
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device set to use: {device}")

# Load model and tokenizer
model_path = "./production_model"  # Path to fine-tuned xlm-roberta-base
model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device=0 if torch.cuda.is_available() else -1)

# Label mapping (from Task 4)
unique_labels = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Load dataset for difficult cases
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

sentences, labels = load_conll_data("cleaned_labeled_data.conll")

# Select test sentences (including difficult cases)
test_sentences = [
    "ጫማዎች 2800 ብር ቦሌ መድኃኔዓለም",  # Clear case
    "ቡና መፍጫ 2100 ብር መገናኛ ወደ ቦሌ",  # Ambiguous location
    "ጫማዎች ቦሌ 2800 ብር",  # Overlapping entities (product and location)
    "TIMBERLAND አዲስ አበባ 7500 ብር",  # Mixed language
]

# SHAP Explainer
def shap_explain(sentence):
    def model_predict(texts):
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**encodings)
        logits = outputs.logits.cpu().numpy()
        return logits

    explainer = Explainer(model_predict, tokenizer)
    shap_values = explainer([sentence])
    logger.info(f"SHAP Values for '{sentence}':")
    for i, token in enumerate(shap_values.data[0]):
        token_contributions = {idx_to_label[j]: shap_values.values[0][i][j] for j in range(len(unique_labels))}
        logger.info(f"Token: {token}, Contributions: {token_contributions}")
    return shap_values

# LIME Explainer
def lime_explain(sentence):
    def model_predict_proba(texts):
        results = []
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**encodings)
            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            # Aggregate subword probabilities (take mean for simplicity)
            aggregated_probs = np.mean(probs, axis=1)
            results.append(aggregated_probs[0])
        return np.array(results)

    explainer = LimeTextExplainer(class_names=unique_labels)
    explanation = explainer.explain_instance(
        sentence,
        model_predict_proba,
        num_features=10,
        labels=range(len(unique_labels))
    )
    logger.info(f"LIME Explanation for '{sentence}':")
    for label_idx in range(len(unique_labels)):
        logger.info(f"Label {unique_labels[label_idx]}:")
        logger.info(explanation.as_list(label=label_idx))
    return explanation

# Analyze difficult cases
def analyze_difficult_cases():
    difficult_cases = test_sentences[1:]  # Exclude clear case
    for sentence in difficult_cases:
        logger.info(f"\nAnalyzing Difficult Case: '{sentence}'")
        # Run NER
        predictions = nlp(sentence)
        logger.info(f"Predictions: {predictions}")
        # SHAP
        shap_values = shap_explain(sentence)
        # LIME
        lime_explanation = lime_explain(sentence)

# Generate interpretability report
def generate_report():
    report = "# NER Model Interpretability Report\n\n"
    report += "## Model: xlm-roberta-base\n"
    report += f"- F1 Score: 0.35 (estimated from Task 4)\n"
    report += f"- Training Time: ~400s\n"
    report += f"- Inference Time: ~0.05s/sentence\n\n"
    
    report += "## SHAP Analysis\n"
    report += "SHAP assigns contributions to each token for each label prediction:\n"
    for sentence in test_sentences:
        report += f"### Sentence: {sentence}\n"
        shap_values = shap_explain(sentence)
        for i, token in enumerate(shap_values.data[0]):
            contributions = {idx_to_label[j]: shap_values.values[0][i][j] for j in range(len(unique_labels))}
            report += f"- Token: {token}, Contributions: {contributions}\n"
    
    report += "\n## LIME Analysis\n"
    report += "LIME provides local feature importance for each label:\n"
    for sentence in test_sentences:
        report += f"### Sentence: {sentence}\n"
        explanation = lime_explain(sentence)
        for label_idx in range(len(unique_labels)):
            report += f"#### Label: {unique_labels[label_idx]}\n"
            report += str(explanation.as_list(label=label_idx)) + "\n"
    
    report += "\n## Difficult Cases Analysis\n"
    for sentence in test_sentences[1:]:
        report += f"### Sentence: {sentence}\n"
        predictions = nlp(sentence)
        report += f"- Predictions: {predictions}\n"
        # Add manual analysis
        if sentence == test_sentences[1]:
            report += "- Issue: Ambiguous location ('መገናኛ ወደ ቦሌ' may confuse model due to multiple location tokens).\n"
        elif sentence == test_sentences[2]:
            report += "- Issue: Overlapping entities ('ጫማዎች ቦሌ' may be misclassified due to proximity).\n"
        elif sentence == test_sentences[3]:
            report += "- Issue: Mixed language ('TIMBERLAND' in English, rest in Amharic) may affect tokenization.\n"
    
    report += "\n## Areas for Improvement\n"
    report += "- **Dataset Quality**: Add more diverse examples to handle ambiguous locations.\n"
    report += "- **Tokenization**: Fine-tune tokenizer for better Amharic subword handling.\n"
    report += "- **Mixed Language**: Preprocess mixed-language inputs to normalize text.\n"
    report += "- **Model Size**: Test xlm-roberta-large for better performance if GPU allows.\n"
    
    with open("interpretability_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to interpretability_report.md")

# Run analysis
if __name__ == "__main__":
    logger.info("Starting interpretability analysis...")
    analyze_difficult_cases()
    generate_report()
    logger.info("Interpretability analysis completed.")
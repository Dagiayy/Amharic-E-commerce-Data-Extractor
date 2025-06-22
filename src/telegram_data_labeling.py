import pandas as pd
import re
import unicodedata
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Amharic tokenizer
def tokenize_amharic(text):
    """
    Tokenize Amharic text, preserving words, numbers, and usernames.
    Returns list of tokens.
    """
    if not text or not isinstance(text, str):
        return []
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('።', '.').replace('፣', ',').replace('፤', ';')
    text = re.sub(r'[^\u1200-\u137F\s0-9@.,;a-zA-Z]', '', text)
    tokens = re.findall(r'[\u1200-\u137F]+|[0-9]+|[@a-zA-Z0-9]+|[.,;]', text)
    tokens = [t for t in tokens if len(t) > 1 or t in ['ኺ', 'ር', '፥', '.', ',', ';', '@']]
    return tokens

# Simple Amharic location dictionary for splitting
LOCATION_DICT = {
    'መገናኛመሰረትደፋርሞልሁለተኛፎቅ': ['መገናኛ', 'መሰረት ደፋር', 'ሁለተኛ ፎቅ'],
    'አዲስአበባ': ['አዲስ አበባ'],
    'ሜክሲኮ': ['ሜክሲኮ'],
    'ኬኬርህንጻ': ['ኬኬር ህንጻ'],
    'አይመንህንፃ': ['አይመን ህንፃ']
}

# Rule-based labeling (to be refined manually)
def label_tokens(tokens):
    """
    Assign NER labels to tokens based on rules.
    Returns list of labels.
    """
    labels = ['O'] * len(tokens)
    for i, token in enumerate(tokens):
        # Product detection (English/Amharic product names)
        if token.lower() in ['saachi', 'nike', 'bottle', 'sneaker', 'humidifier', 'carrier'] or re.match(r'[\u1200-\u137F]+', token) in ['ኬትል', 'ስኒከር']:
            labels[i] = 'B-Product'
            j = i + 1
            while j < len(tokens) and (tokens[j].lower() in ['electric', 'kettle', 'stopper', 'crease', 'protector', 'air', 'force', 'usb', 'ultrasonic'] or re.match(r'[\u1200-\u137F]+', tokens[j])):
                labels[j] = 'I-Product'
                j += 1
        # Price detection
        if token in ['ዋጋ', 'Price', 'በ']:
            if i + 1 < len(tokens) and re.match(r'[0-9]+', tokens[i + 1]):
                labels[i + 1] = 'B-PRICE'
                if i + 2 < len(tokens) and tokens[i + 2] in ['ብር', 'Br']:
                    labels[i + 2] = 'I-PRICE'
        # Location detection
        if token in ['አድራሻ', 'መገናኛ', 'አዲስ', 'ሜክሲኮ', 'ኬኬር', 'አይመን']:
            labels[i] = 'B-LOC'
            j = i + 1
            while j < len(tokens) and tokens[j] in ['አበባ', 'መሰረት', 'ደፋር', 'ሁለተኛ', 'ፎቅ', 'ህንጻ', 'ህንፃ']:
                labels[j] = 'I-LOC'
                j += 1
        # Compound location splitting
        for compound, split in LOCATION_DICT.items():
            if ' '.join(tokens[i:i + len(split)]) == compound.replace(' ', ''):
                for j, subtoken in enumerate(split):
                    labels[i + j] = 'B-LOC' if j == 0 else 'I-LOC'
    return labels

# Main function
def main():
    # Load data
    try:
        df = pd.read_csv('data/telegram_data.csv')
        logger.info(f"Loaded {len(df)} rows from telegram_data.csv")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return

    # Sample 30 messages with non-empty combined_text
    df = df[df['combined_text'].str.strip().ne('')].sample(n=min(30, len(df)), random_state=42)
    logger.info(f"Selected {len(df)} messages for labeling")

    # Save sample messages for manual review
    df[['combined_text', 'channel']].to_csv('sample_messages.csv', index=False, encoding='utf-8')
    logger.info("Saved sample messages to sample_messages.csv")

    # Label messages
    labeled_data = []
    for _, row in df.iterrows():
        text = row['combined_text']
        # Split compound locations
        for compound, split in LOCATION_DICT.items():
            text = text.replace(compound, ' '.join(split))
        tokens = tokenize_amharic(text)
        if not tokens:
            logger.warning(f"Empty tokens for message: {text[:50]}...")
            continue
        labels = label_tokens(tokens)
        labeled_data.append((tokens, labels))
        logger.debug(f"Labeled {len(tokens)} tokens for message: {text[:50]}...")

    # Save to CoNLL format
    try:
        with open('labeled_data.conll', 'w', encoding='utf-8') as f:
            for tokens, labels in labeled_data:
                for token, label in zip(tokens, labels):
                    f.write(f"{token} {label}\n")
                f.write("\n")
        logger.info("Saved labeled data to labeled_data.conll")
    except Exception as e:
        logger.error(f"Error saving CoNLL file: {e}")

if __name__ == '__main__':
    main()
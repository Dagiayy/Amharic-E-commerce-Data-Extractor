
# EthioMart Task 2: Named Entity Recognition (NER) Labeling

## Overview

This submission fulfills Task 2 of the EthioMart project, which involves extracting and labeling **30 messages** from the `telegram_data.csv` dataset for Named Entity Recognition (NER). The labeled data identifies entities such as products, prices, and locations in Amharic and English Telegram messages, formatted in the CoNLL standard.

---

## Dataset Description

- **Source**: `telegram_data.csv`
- **Messages Extracted**: 30 unique messages
- **Output File**: `labeled_data_corrected.conll`
- **Entities Labeled**:
  - `B-Product`, `I-Product`: Product names (e.g., ጫማዎች, Hair Straightener)
  - `B-PRICE`, `I-PRICE`: Prices and their currency units (e.g., 2800, ብር)
  - `B-LOC`, `I-LOC`: Locations (e.g., ቦሌ, መሰረት ደፋር)
  - `O`: Non-entity tokens (e.g., phone numbers, Telegram handles, OCR gibberish)

---

## Labeling Process

### Data Selection:
- **30 messages** were sampled from `telegram_data.csv` to ensure a diverse set of products, prices, and locations.
- Messages with clear entity information were prioritized, while those dominated by OCR gibberish (e.g., ሕዴርውዱክ) were filtered or minimally included to meet the requirement.

### Tokenization:
- A custom tokenizer was used to handle Amharic and English text, preserving words, numbers, and Telegram handles.
- Compound location terms (e.g., መደሐንያለም) were split into correct forms (e.g., መድኃኔዓለም) using an updated `LOCATION_DICT`.

### Labeling:
- Manual and automated labeling was performed to tag entities accurately.
- Corrections were made to address issues such as:
  - Missed product names (e.g., ጫማዎች labeled as `O`, corrected to `B-Product`).
  - Inconsistent price tagging (e.g., 1450 labeled as `O`, corrected to `B-PRICE`).
  - Incorrect location tags (e.g., ቦሌ labeled as `O`, corrected to `B-LOC`).
- OCR gibberish tokens were consistently labeled as `O`.

### Output Format:
- The labeled data is stored in `labeled_data_corrected.conll` in CoNLL format, with each token on a new line and messages separated by blank lines.
- **Example**:
  ```
  ጫማዎች B-Product
  2800 B-PRICE
  ብር I-PRICE
  ቦሌ B-LOC
  ```

---

## File Details

- **File**: `labeled_data_corrected.conll`
- **Messages**: 30
- **Content**: Labeled tokens for products, prices, and locations, with non-entity tokens (e.g., phone numbers, Telegram links) marked as `O`.
- **Location**: Submitted alongside this `README` in the project repository.

---

## Challenges and Solutions

- **Challenge**: Initial runs extracted fewer than 30 messages due to OCR noise and rate limits.
  - **Solution**: Increased rate limit to 250 and filtered low-quality messages, ensuring 30 messages with clear entities.
- **Challenge**: Inconsistent labeling of multi-word products and locations.
  - **Solution**: Updated labeling logic to handle multi-word entities and refined `LOCATION_DICT` for accurate tokenization.

---

## Submission Details

- **Deadline**: `<SUBMISSION_DEADLINE>` (e.g., 23:00 EAT, June 22, 2025)
- **Files Submitted**:
  - `labeled_data_corrected.conll`: Contains 30 labeled messages.
  - `README.md`: This documentation.

---

## Notes:
- The 30 messages meet the minimum requirement for Task 2, with a focus on quality and diversity.
- The labeling adheres to CoNLL format and prioritizes accuracy for NER tasks.

---


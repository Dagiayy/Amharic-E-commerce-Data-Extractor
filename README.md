# Amharic-E-commerce-Data-Extractor
EthioMart Task 1: Telegram Data Ingestion Pipeline
Overview
This repository contains the implementation for Task 1 of the EthioMart project, a data ingestion pipeline that collects messages from Ethiopian Telegram e-commerce channels, preprocesses Amharic text, extracts text from images using OCR, and structures the data for Named Entity Recognition (NER). The pipeline targets entities such as products, prices, and locations to support the development of an e-commerce search engine.
The script (telegram_data_ingestion_optimized.py) successfully fetched and processed messages from two channels (t.me/ZemenExpress and t.me/marakibrand), producing a CSV with 34 rows of structured data. This README provides instructions for setup, usage, and details about the output and challenges.
Prerequisites

Python: Version 3.8+
Dependencies:pip install telethon pandas tesserocr Pillow


Tesseract OCR:
Install Tesseract with Amharic support (amh.traineddata).
Windows path: C:/Program Files/Tesseract-OCR/tessdata.
Verify: tesseract --list-langs (should include amh).


Telegram API Credentials:
api_id: 25137965
api_hash: 5b75b63d388dc496776c90b42da6f4e6
phone: +251967909687 (requires login code and possibly 2FA password).



Setup

Clone the repository:git clone <repository-url>
cd <repository-directory>


Install dependencies:pip install -r requirements.txt


Ensure Tesseract is installed and amh.traineddata is in TESSDATA_PATH.
Create a virtual environment (optional):python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate



Usage

Run the script:python telegram_data_ingestion_optimized.py


Enter the Telegram login code sent to +251967909687. Provide the 2FA password if prompted.
The script:
Connects to Telegram channels.
Fetches up to 100 messages per channel.
Downloads media (images/documents).
Extracts text from images using Tesseract OCR.
Preprocesses Amharic text with a custom tokenizer.
Saves structured data to data/telegram_data.csv.



Pipeline Description

Connection:
Initializes a TelegramClient using Telethon.
Verifies access to channels: t.me/ZemenExpress, t.me/Leyueqa, t.me/helloomarketethiopia, t.me/kuruwear, t.me/marakibrand.


Message Fetching:
Collects messages (text, images, documents) with metadata (sender, timestamp, views).


Image Preprocessing:
Converts images to grayscale and enhances contrast for better OCR accuracy.


OCR:
Uses Tesseract with lang='amh' and PSM.SINGLE_BLOCK to extract Amharic text from images.


Text Preprocessing:
Normalizes Unicode, standardizes punctuation, and removes non-Amharic characters.
Tokenizes text, preserving Amharic words, numbers, usernames, and punctuation.
Filters invalid tokens (e.g., single characters, except valid ones like ኺ, ር).


Data Structuring:
Combines text and image_text into combined_text.
Filters rows with non-empty content.
Saves to a pandas DataFrame with columns: channel, sender, timestamp, views, text, media_type, media, image_text, combined_text.


Output:
Saves DataFrame to data/telegram_data.csv.



Output

File: data/telegram_data.csv
Rows: 34
Channels:
t.me/ZemenExpress: 15 rows
t.me/marakibrand: 19 rows


Columns:
channel: Telegram channel URL
sender: Sender ID
timestamp: Message date/time
views: View count
text: Message text
media_type: Image or document
media: BytesIO object
image_text: OCR-extracted text
combined_text: Concatenated text and image_text


Sample Row:channel,sender,timestamp,views,text,media_type,media,image_text,combined_text
t.me/ZemenExpress,-1001307493052,2025-06-21 16:35:51+00:00,1878,". . . Saachi Electric Kettle Borosilicate Glass Body ... ዋጋ፦ 2700 ብር ... አድራሻ መገናኛመሰረትደፋርሞልሁለተኛፎቅ ... @zemencallcenter ...",image,<_io.BytesIO object>,"ጻኋሺዷ ፦፦ ;",". . . Saachi Electric Kettle ... ዋጋ፦ 2700 ብር ... መገናኛመሰረትደፋርሞልሁለተኛፎቅ ... @zemencallcenter ... ጻኋሺዷ ፦፦ ;"


Usage: The CSV supports Task 2 (labeling 30–50 messages in CoNLL format for NER).

Challenges

OCR Accuracy:
Issue: image_text often contains gibberish (e.g., ሃፍከር, ጻኋሺዷ).
Cause: Low-quality images or complex layouts.
Mitigation: Added image preprocessing (grayscale, contrast enhancement) and PSM.SINGLE_BLOCK.


Data Volume:
Issue: Only 34 rows from 2/5 channels.
Cause: Inaccessible channels or sparse recent messages.
Mitigation: Increased fetch limit to 100; planned retry logic and older message fetching.


Compound Words:
Issue: Locations like መገናኛመሰረትደፋርሞልሁለተኛፎቅ are not split.
Mitigation: Implemented basic dictionary-based splitting; full splitting requires an Amharic dictionary.


Filtering:
Issue: Many messages filtered out due to empty text/image_text.
Mitigation: Relaxed filtering to retain rows with minimal valid content.



Future Improvements

Enhance OCR with sharpening, thresholding, and confidence score filtering.
Implement advanced compound word splitting using an Amharic lexicon.
Add retry logic for rate limits and fetch older messages.
Test additional channels for broader data coverage.

Task 2 Preparation

Use telegram_data_labeling.py to sample 30 rows from telegram_data.csv.
Label entities (B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O) in CoNLL format.
Example:Saachi B-Product
Electric I-Product
Kettle I-Product
ዋጋ O
2700 B-PRICE
ብር I-PRICE
መገናኛ B-LOC



Contact
For issues or questions, please contact the project team via GitHub issues or Telegram.
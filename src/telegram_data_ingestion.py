import asyncio
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError
import pandas as pd
import re
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image, ImageEnhance
import io
import unicodedata
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram API credentials
api_id = 25137965
api_hash = '5b75b63d388dc496776c90b42da6f4e6'
phone = '+251967909687'

# Ethiopian Telegram e-commerce channels
channels = [
    't.me/ZemenExpress',
    't.me/Leyueqa',
    't.me/helloomarketethiopia',
    't.me/kuruwear',
    't.me/marakibrand'
]

# Tesseract tessdata path (for Windows)
TESSDATA_PATH = 'C:/Program Files/Tesseract-OCR/tessdata'

# Step 1: Connect to Telegram
async def connect_to_telegram():
    """
    Initialize Telegram client and verify connection to channels.
    Returns the client object and list of accessible channels.
    """
    try:
        client = TelegramClient('session_name', api_id, api_hash)
        await client.start(phone=phone)
        logger.info("Telegram client initialized successfully")
        accessible_channels = []
        for channel in channels:
            try:
                entity = await client.get_entity(channel)
                logger.info(f"Successfully connected to {channel} (Title: {entity.title})")
                accessible_channels.append(channel)
            except Exception as e:
                logger.error(f"Error connecting to {channel}: {e}")
        if not accessible_channels:
            raise Exception("No accessible channels")
        return client, accessible_channels
    except SessionPasswordNeededError:
        logger.error("Two-factor authentication required. Please provide password.")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Telegram client: {e}")
        raise

# Step 2: Fetch messages
async def fetch_telegram_messages(client, accessible_channels):
    """
    Fetch messages, including text, images, and documents, from accessible channels.
    Returns a list of message dictionaries.
    """
    messages = []
    for channel in accessible_channels:
        try:
            channel_messages = []
            async for message in client.iter_messages(channel, limit=250):  # Increased limit
                msg_data = {
                    'channel': channel,
                    'sender': message.sender_id if message.sender_id else 0,
                    'timestamp': message.date,
                    'views': message.views if message.views else 0,
                    'text': message.text if message.text else '',
                    'media_type': None,
                    'media': None
                }
                if message.media:
                    try:
                        if hasattr(message.media, 'photo'):
                            msg_data['media_type'] = 'image'
                            msg_data['media'] = await client.download_media(message.media, file=io.BytesIO())
                        elif hasattr(message.media, 'document'):
                            msg_data['media_type'] = 'document'
                            msg_data['media'] = await client.download_media(message.media, file=io.BytesIO())
                        logger.info(f"Downloaded media from {channel} at {message.date}")
                    except Exception as e:
                        logger.error(f"Error downloading media from {channel}: {e}")
                channel_messages.append(msg_data)
            logger.info(f"Fetched {len(channel_messages)} messages from {channel}")
            messages.extend(channel_messages)
        except Exception as e:
            logger.error(f"Error fetching messages from {channel}: {e}")
    return messages

# Step 3: Preprocess image for OCR
def preprocess_image(image):
    """
    Enhance image for better OCR accuracy.
    Returns preprocessed image.
    """
    try:
        image = image.convert('L')  # Grayscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Increase contrast
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

# Step 4: Preprocess Amharic text
def preprocess_amharic_text(text):
    """
    Preprocess Amharic text by normalizing, removing extra spaces, and tokenizing.
    Returns cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""
    try:
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        # Standardize spaces
        text = re.sub(r'\s+', ' ', text)
        # Standardize punctuation
        text = text.replace('።', '.').replace('፣', ',').replace('፤', ';')
        # Remove non-Amharic characters (keep numbers, @ for usernames)
        text = re.sub(r'[^\u1200-\u137F\s0-9@.,;a-zA-Z]', '', text)
        # Custom Amharic tokenizer
        tokens = re.findall(r'[\u1200-\u137F]+|[0-9]+|[@a-zA-Z0-9]+|[.,;]', text)
        # Filter invalid tokens
        tokens = [t for t in tokens if len(t) > 1 or t in ['ኺ', 'ር', '፥', '.', ',', ';', '@']]
        cleaned_text = ' '.join(tokens).strip()
        logger.debug(f"Preprocessed text: {cleaned_text[:50]}...")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return ""

# Step 5: Extract text from images
def extract_text_from_image(media, media_type):
    """
    Extract text from images using Tesseract OCR with Amharic support.
    Returns extracted text.
    """
    if not media or media_type != 'image':
        return ""
    try:
        media_bytes = media.getvalue() if isinstance(media, io.BytesIO) else media
        image = Image.open(io.BytesIO(media_bytes))
        image = preprocess_image(image)
        with PyTessBaseAPI(lang='amh', psm=PSM.SINGLE_BLOCK, path=TESSDATA_PATH) as api:
            api.SetImage(image)
            extracted_text = api.GetUTF8Text()
            logger.debug(f"Raw OCR text: {extracted_text[:50]}...")
            logger.info("Extracted text from image")
            cleaned_text = preprocess_amharic_text(extracted_text)
            logger.debug(f"Extracted image text: {cleaned_text[:50]}...")
            return cleaned_text
    except Exception as e:
        logger.error(f"Error in OCR: {e}")
        return ""

# Step 6: Structure data
def structure_data(messages):
    """
    Combine text and image data, clean metadata, and structure into a DataFrame.
    Returns a pandas DataFrame.
    """
    try:
        df = pd.DataFrame(messages)
        logger.info(f"Raw DataFrame has {len(df)} rows")
        df['text'] = df['text'].apply(preprocess_amharic_text)
        df['image_text'] = df.apply(lambda row: extract_text_from_image(row['media'], row['media_type']), axis=1)
        df['combined_text'] = df['text'] + ' ' + df['image_text']
        df['sender'] = df['sender'].fillna(0).astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['views'] = df['views'].fillna(0).astype(int)
        # Log rows to be filtered
        filtered_out = df[(df['text'].str.strip() == '') & (df['image_text'].str.strip() == '')]
        if not filtered_out.empty:
            logger.debug(f"Filtering out {len(filtered_out)} rows with empty text and image_text")
        # Keep rows with any content
        df = df[(df['text'].str.strip() != '') | (df['image_text'].str.strip() != '')]
        logger.info(f"Filtered DataFrame has {len(df)} rows")
        # Log sample rows
        if not df.empty:
            logger.debug(f"Sample combined_text: {df['combined_text'].iloc[0][:50]}...")
        return df
    except Exception as e:
        logger.error(f"Error structuring data: {e}")
        return pd.DataFrame()

# Step 7: Save data
def save_data(df):
    """
    Save preprocessed data to a CSV file in the 'data' directory.
    """
    try:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/telegram_data.csv', index=False, encoding='utf-8')
        logger.info("Saved data to data/telegram_data.csv")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

# Main execution
async def main():
    client, accessible_channels = await connect_to_telegram()
    messages = await fetch_telegram_messages(client, accessible_channels)
    logger.info(f"Fetched {len(messages)} messages")
    preprocessed_df = structure_data(messages)
    if not preprocessed_df.empty:
        save_data(preprocessed_df)
    else:
        logger.error("No data to save")
    return preprocessed_df

# Run the async main
if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"Main execution error: {e}")
import pandas as pd
import numpy as np
import re
import sys, os
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob  # For spelling correction
import emoji

from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath('src'))
from utility.utility import load_params

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Precompiled stopwords set for faster lookups
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# To skip spell correction of words in the valid_words set
valid_words = []

# Dictionary for contractions
contractions_dict = {
    "can't": "cannot",
    "won't": "will not",
    "I'm": "I am",
    "it's": "it is",
    "don't": "do not",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "we're": "we are",
    "I've": "I have",
    "you've": "you have",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    # Add more as needed
}

def expand_contractions(text):
    """Replace contractions with their full forms."""
    for contraction, full_form in contractions_dict.items():
        text = re.sub(rf'\b{contraction}\b', full_form, text)
    return text

def translate_emojis(text):
    """Detect and replace emojis with their corresponding textual descriptions."""
    text =  emoji.demojize(text, delimiters=(" ", " "))
    return text.replace('_', ' ')

def remove_unwanted_characters(text):
    """Remove all characters except English, Devanagari, apostrophes, and spaces."""
    return re.sub(r"[^\u0900-\u097Fa-zA-Z\s']", '', text)

def remove_newlines(text):
    """Replace newlines with spaces."""
    return text.replace('\n', ' ')

def compress_spaces(text):
    """Replace multiple spaces with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def limit_repeated_characters(text):
    """Limit repeated characters to a maximum of two."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def lemmatization(text):
    """Lemmatize the text."""
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    """Remove stop words from the text."""
    return " ".join([word for word in text.split() if word not in stop_words])


def removing_numbers(text):
    """Remove numbers from the text."""
    return ''.join(char for char in text if not char.isdigit())

def lower_case(text):
    """Convert text to lower case."""
    return text.lower()

def removing_punctuations(text):
    """Remove punctuations from the text."""
    return re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text).strip()

def removing_urls(text):
    """Remove URLs from the text."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def spelling_correction(text):
    """Perform spelling correction on English words only, skipping valid words."""
    text = limit_repeated_characters(text)
    corrected_text = []
    for word in text.split():
        # Skip spell correction for words in the valid_words set
        if word.lower() in valid_words:
            corrected_text.append(word)
        else:
            if not re.search(r'[\u0900-\u097F]', word):
                # Correct English words only
                corrected_word = str(TextBlob(word).correct())
                corrected_text.append(corrected_word)
            else:
                corrected_text.append(word)
    return ' '.join(corrected_text)

def process_text(df):
    """Normalize the text data, including emoji translation."""
    df['content'] = (df['clean_comment']
                     .apply(lower_case)  # Step 1: Convert to lower case
                     .apply(translate_emojis)  # Step 2: Translate emojis to text
                     .apply(expand_contractions)  # Step 3: Replace contractions
                     .apply(remove_unwanted_characters)  # Step 4: Remove unwanted characters
                     .apply(removing_numbers)  # Step 5: Remove numbers
                     .apply(removing_punctuations)  # Step 6: Remove punctuation (apostrophes kept)
                     .apply(removing_urls)  # Step 7: Remove URLs
                     .apply(remove_newlines)  # Step 8: Remove newlines
                     .apply(limit_repeated_characters)  # Step 9: Limit repeated characters
                     .apply(remove_stop_words)  # Step 10: Remove stop words
                     .apply(lemmatization)  # Step 11: Lemmatization
                     .apply(spelling_correction)  # Step 12: Spell correction for English
                     .apply(compress_spaces))  # Step 13: Ensure only single space
    return df

def concat_text_features(df, text_features):
    df['text_feature'] = df[text_features].agg(' '.join, axis=1)
    return df

def main():
    try:
        # Fetch the data from data/raw
        data = pd.read_csv('./data/raw/data.csv')

        # Loading params for
        params = load_params()
        text_features = params['columns']['features']
        target = params['columns']['target']
        test_size = params['data_ingestion']['test_size']
        val_size = params['data_ingestion']['val_size'] 
        random_state = params['data_ingestion']['random_state']
        data.dropna(subset=text_features, inplace=True)
        data = concat_text_features(data, text_features)

        processed_data = process_text(data)
        temp_data, test_processed_data = train_test_split(processed_data, test_size=test_size, stratify=processed_data[target], random_state=random_state)
        train_processed_data, val_processed_data = train_test_split(temp_data, test_size=val_size, stratify=temp_data[target], random_state=random_state)
        
        print("Processed Data")
        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        val_processed_data.to_csv(os.path.join(data_path, "val_processed.csv"), index=False)
        
    except Exception as e:
        print('Failed to complete the data transformation process: %s')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    print("Processing complete")

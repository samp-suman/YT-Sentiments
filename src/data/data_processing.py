import pandas as pd
import numpy as np
import re
import sys, os, json, requests
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

# Define the path for the cache file
params = load_params()
cache_url = params['data_source']['cache_file_url']


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
    text = re.sub(r"[^\u0900-\u097Fa-zA-Z\s']", '', text)
    return text

def remove_newlines(text):
    """Replace newlines with spaces."""
    text = text.replace('\n', ' ')
    return text

def compress_spaces(text):
    """Replace multiple spaces with a single space."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def limit_repeated_characters(text):
    """Limit repeated characters to a maximum of two."""
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

def lemmatization(text):
    """Lemmatize the text."""
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def remove_stop_words(text):
    """Remove stop words from the text."""
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join(char for char in text if not char.isdigit())
    return text

def lower_case(text):
    """Convert text to lower case."""
    return text.lower()

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

# Load Cache to store already corrected words
response = requests.get(cache_url)
corrected_cache = response.json() if response.status_code == 200 else {}
cache_file_path = "data/interim/spelling_correction_cache.json"

def spelling_correction(text):
    """Perform spelling correction on English words only, skipping valid words and using a cache for optimization."""
    
    # Limit repeated characters first to minimize unnecessary corrections
    text = limit_repeated_characters(text)
    corrected_text = []

    for word in text.split():
        # Skip spell correction for words in the valid_words set or words with less than 3 characters
        if word.lower() in valid_words or len(word) < 3:
            corrected_text.append(word)
        else:
            # Use cached corrections if available
            if word in corrected_cache:
                corrected_text.append(corrected_cache[word])
            else:
                if not re.search(r'[\u0900-\u097F]', word):  # Correct only English words
                    corrected_word = str(TextBlob(word).correct())
                    corrected_cache[word] = corrected_word  # Cache the result
                    corrected_text.append(corrected_word)
                else:
                    corrected_text.append(word)

    # Save the cache to a JSON file at the end of processing
    with open(cache_file_path, 'w') as cache_file:
        json.dump(corrected_cache, cache_file)

    return ' '.join(corrected_text)



def process_text(df):
    """Normalize the text data, including emoji translation and URL handling."""
    
    # Step 1: Convert to lower case
    df['content'] = df['text_feature'].apply(lower_case)
    print("Step 1: Lowercasing - Done")

    # Step 2: Translate emojis to text
    df['content'] = df['content'].apply(translate_emojis)
    print("Step 2: Emoji translation - Done")

    # Step 3: Handle URLs before removing unwanted characters
    df['content'] = df['content'].apply(removing_urls)
    print("Step 3: URL removal - Done")

    # Step 4: Replace contractions
    df['content'] = df['content'].apply(expand_contractions)
    print("Step 4: Contractions expansion - Done")

    # Step 5: Remove unwanted characters (punctuation and newlines already handled)
    df['content'] = df['content'].apply(remove_unwanted_characters)
    print("Step 5: Unwanted character removal - Done")

    # Step 6: Remove stop words
    df['content'] = df['content'].apply(remove_stop_words)
    print("Step 6: Stop word removal - Done")

    # Step 7: Lemmatization
    df['content'] = df['content'].apply(lemmatization)
    print("Step 7: Lemmatization - Done")

    # Step 8: Spell correction for English
    df['content'] = df['content'].apply(spelling_correction)
    print("Step 8: Spell correction - Done")

    # Step 9: Ensure only single space
    df['content'] = df['content'].apply(compress_spaces)
    print("Step 9: Space compression - Done")

    # Step 10: Remove Nan - if any introduced due to cleanig
    df.dropna(subset=['content'], inplace=True)

    return df[['text_feature', 'content', 'category']]

def concat_text_features(df, text_features):
    df['text_feature'] = df[text_features].agg(' '.join, axis=1)
    return df

def main():
    try:
        # Fetch the data from data/raw
        data = pd.read_csv('./data/raw/data.csv')

        # Load parameters
        text_features = params['columns']['features']
        target = params['columns']['target']
        test_size = params['data_ingestion']['test_size']
        val_size = params['data_ingestion']['val_size'] 
        random_state = params['data_ingestion']['random_state']
        
        # Dropping rows with missing text data 
        data.dropna(subset=text_features, how='all', inplace=True)

        # If we have multiple text columns, we need to concatenate them
        # text_features is a list of text column names from params.yml
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

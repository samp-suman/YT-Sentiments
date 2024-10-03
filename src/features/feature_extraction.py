import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.abspath('src'))
from utility.utility import load_params


# (Include your existing text processing functions here...)

def extract_bow_features(df_train, df_test, df_val, max_features, ngram_range):
    """Create Bag of Words features."""
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    train_bow = vectorizer.fit_transform(df_train['content'])
    test_bow = vectorizer.transform(df_test['content'])
    val_bow = vectorizer.transform(df_val['content'])
    return train_bow, test_bow, val_bow, vectorizer.get_feature_names_out()


def main():
    try:
        # Fetch the data from data/interim
        train_data = pd.read_csv('./data/interim/train_processed.csv')
        test_data = pd.read_csv('./data/interim/test_processed.csv')
        val_data = pd.read_csv('./data/interim/val_processed.csv')

        print(train_data.info())
        print(test_data.info())
        print(val_data.info())

        print(train_data[train_data['content'].isna()])
        # Load parameters
        params = load_params()
        bow_params = params['feature_extraction']['bow']

        # Apply Bag of Words approach
        max_features = bow_params['max_features']
        ngram_range = tuple(bow_params['ngram_range'])
        train_bow, test_bow, val_bow, feature_names = extract_bow_features(train_data, test_data, val_data, max_features, ngram_range)
        
        train_features_df = pd.DataFrame(train_bow.toarray(), columns=feature_names)
        test_features_df = pd.DataFrame(test_bow.toarray(), columns=feature_names)
        val_features_df = pd.DataFrame(val_bow.toarray(), columns=feature_names)

        train_features_df['target_label'] = train_data['category']
        test_features_df['target_label'] = test_data['category']
        val_features_df['target_label'] = val_data['category']

        # Store the data inside data/processed
        train_features_df.to_csv(os.path.join('./data/processed', 'train_bow_features.csv'), index=False)
        test_features_df.to_csv(os.path.join('./data/processed', 'test_bow_features.csv'), index=False)
        val_features_df.to_csv(os.path.join('./data/processed', 'val_bow_features.csv'), index=False)
        
    
        print("Feature extraction complete")

    except Exception as e:
        print(f'Failed to complete the feature extraction process: {e}')

if __name__ == '__main__':
    main()

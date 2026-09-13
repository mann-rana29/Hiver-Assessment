import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATASET_PATH = os.path.join(DATA_DIR, 'twcs', 'twcs.csv')
GOLDEN_SET_TEMPLATE_PATH = os.path.join(DATA_DIR, 'golden_set_template.csv')
BRAND_NAME = 'AppleSupport'
NUM_SAMPLES = 500

def download_and_extract_data():
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}. Skipping download.")
        return

    print("Downloading dataset from Kaggle...")
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('thoughtvector/customer-support-on-twitter', path=DATA_DIR, unzip=True)
    
    # The unzipped file might be in a nested directory depending on how Kaggle unzips it.
    # We assume twcs.csv is placed inside DATA_DIR.
    print("Download and extraction complete.")

def process_data():
    print("Loading data into pandas (this might take a minute as it's a large file)...")
    df = pd.read_csv(DATASET_PATH, dtype={'tweet_id': str, 'in_response_to_tweet_id': str, 'author_id': str})
    
    print(f"Filtering for {BRAND_NAME}...")
    
    # Tweets by the brand
    brand_tweets = df[df['author_id'] == BRAND_NAME].copy()
    
    # Find the tweets the brand responded to
    brand_responded_to_ids = brand_tweets['in_response_to_tweet_id'].dropna().astype(str).tolist()
    
    # Customer tweets
    df['tweet_id_str'] = df['tweet_id'].astype(str)
    customer_tweets = df[df['tweet_id_str'].isin(brand_responded_to_ids)].copy()
    
    # Rename columns for clarity
    customer_tweets = customer_tweets.rename(columns={'text': 'customer_message', 'tweet_id_str': 'customer_tweet_id'})
    brand_tweets['in_response_to_tweet_id_str'] = brand_tweets['in_response_to_tweet_id'].astype(str)
    brand_tweets = brand_tweets.rename(columns={'text': 'brand_response', 'in_response_to_tweet_id_str': 'customer_tweet_id'})
    
    # Merge to form pairs
    conversations = pd.merge(
        customer_tweets[['customer_tweet_id', 'customer_message', 'author_id', 'created_at']], 
        brand_tweets[['customer_tweet_id', 'brand_response']], 
        on='customer_tweet_id', 
        how='inner'
    )
    
    # Remove duplicates if any (e.g., if a brand replied twice)
    conversations = conversations.drop_duplicates(subset=['customer_tweet_id'])
                             
    print(f"Total conversations found: {len(conversations)}")
    
    print(f"Sampling {min(NUM_SAMPLES, len(conversations))} conversations...")
    sampled_conversations = conversations.sample(n=min(NUM_SAMPLES, len(conversations)), random_state=42)
    
    # Add empty columns for manual labeling
    sampled_conversations['intent'] = ''
    sampled_conversations['auto_handle'] = ''
    sampled_conversations['escalate_reason'] = ''
    
    sampled_conversations.to_csv(GOLDEN_SET_TEMPLATE_PATH, index=False)
    print(f"Saved golden set template to {GOLDEN_SET_TEMPLATE_PATH}")
    print("Please review and manually label 150-250 of these rows to create data/golden_set.csv")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_and_extract_data()
    process_data()

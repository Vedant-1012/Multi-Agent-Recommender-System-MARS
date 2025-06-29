import pandas as pd
import numpy as np
import time
import os

# --- Configuration ---
# Point to your project's data directories
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

def create_processed_data():
    """
    This function combines the logic from the two notebook cells to load,
    preprocess, and save the final train/test dataframes.
    """
    print("Starting Data Loading and Preprocessing...")

    # --- Create processed data directory if it doesn't exist ---
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")

    # =================================================================
    # START: Code from Notebook Cell 1
    # =================================================================
    start_time = time.time()

    ratings_dtype = {
        'userId': np.int32,
        'movieId': np.int32,
        'rating': np.float32
    }
    movies_dtype = {
        'movieId': np.int32,
        'title': str,
        'genres': str
    }

    try:
        ratings_path = os.path.join(RAW_DATA_DIR, 'rating.csv')
        movies_path = os.path.join(RAW_DATA_DIR, 'movie.csv')
        
        ratings_df = pd.read_csv(ratings_path, dtype=ratings_dtype)
        movies_df = pd.read_csv(movies_path, dtype=movies_dtype)

        print("\nConverting timestamp column to datetime objects...")
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], errors='coerce') #

        print(f"\nLoaded {len(ratings_df)} ratings and {len(movies_df)} movies.")

        print("\nMerging ratings and movies dataframes...")
        df = pd.merge(ratings_df, movies_df, on='movieId') #

    except FileNotFoundError as e:
        print("="*50)
        print(f"ERROR: Dataset file not found. Ensure 'rating.csv' and 'movie.csv' are in '{RAW_DATA_DIR}'")
        print(e)
        print("="*50)
        return
    
    # =================================================================
    # START: Code from Notebook Cell 2
    # =================================================================
    print("\nContinuing with Preprocessing and Feature Engineering...")

    df.dropna(inplace=True) #
    print(f"\nShape after dropping any missing values: {df.shape}")

    print("\nPerforming feature engineering...")
    df['liked'] = (df['rating'] >= 3.5).astype(int) #
    df['year'] = df['title'].str.extract(r'\((\d{4})\)$', expand=False) #
    df['year'] = pd.to_numeric(df['year'], errors='coerce') #

    print("\nSplitting data into training and testing sets based on timestamp...")
    df_sorted = df.sort_values(by='timestamp', ascending=True) #
    
    split_index = int(len(df_sorted) * 0.8) #

    train_df = df_sorted.iloc[:split_index].reset_index(drop=True)
    test_df = df_sorted.iloc[split_index:].reset_index(drop=True)

    print("\nData splitting complete.")
    print(f"Training set shape: {train_df.shape}") #
    print(f"Testing set shape:  {test_df.shape}") #

    # --- Final Step: Save the processed data ---
    train_save_path = os.path.join(PROCESSED_DATA_DIR, 'train_df.csv')
    test_save_path = os.path.join(PROCESSED_DATA_DIR, 'test_df.csv')

    print(f"\nSaving processed dataframes to '{PROCESSED_DATA_DIR}'...")
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    end_time = time.time()
    print(f"\nProcess finished successfully in {end_time - start_time:.2f} seconds.")
    print(f"Files saved:\n- {train_save_path}\n- {test_save_path}")

# This allows the script to be run directly from the command line
if __name__ == '__main__':
    create_processed_data()
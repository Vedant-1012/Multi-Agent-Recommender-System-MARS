import pandas as pd
import numpy as np
import os
import joblib
import time

try:
    from lightfm import LightFM
    from lightfm.data import Dataset
except ImportError:
    print("="*70)
    print("ERROR: The 'lightfm' library is not installed.")
    print("Please install it by running the following command in your terminal:")
    print("\npip install lightfm\n")
    print("="*70)
    exit()

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
ARTIFACTS_DIR = 'artifacts' # A new directory to store our trained models

def train_bpr_model():
    """
    Loads the processed training data and trains a LightFM BPR model,
    then saves the model and dataset mappings to disk.
    """
    print("Starting Step 3.1: Training BPR Baseline Model...")
    start_time = time.time()

    # --- Create artifacts directory if it doesn't exist ---
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        print(f"Created directory: {ARTIFACTS_DIR}")

    # --- Step 1: Load Processed Training Data ---
    train_data_path = os.path.join(PROCESSED_DATA_DIR, 'train_df.csv')
    try:
        print(f"Loading training data from: {train_data_path}")
        train_df = pd.read_csv(train_data_path)
        print("Training data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Training data not found at '{train_data_path}'.")
        print("Please ensure 'data_loading.py' has been run successfully.")
        return

    # --- Step 2: Build LightFM Dataset ---
    # The Dataset object maps our raw user/item IDs to internal integer indices
    print("Building LightFM dataset and interaction matrix...")
    dataset = Dataset()
    
    # We must fit the dataset on all users and items it might see
    # For a fair evaluation later, we'll fit on both train and test users/items
    test_data_path = os.path.join(PROCESSED_DATA_DIR, 'test_df.csv')
    test_df = pd.read_csv(test_data_path)
    all_users = pd.concat([train_df['userId'], test_df['userId']]).unique()
    all_items = pd.concat([train_df['movieId'], test_df['movieId']]).unique()
    
    dataset.fit(users=all_users, items=all_items)
    
    # Build the interaction matrix only from the training data
    (interactions, weights) = dataset.build_interactions(
        (row['userId'], row['movieId']) for _, row in train_df.iterrows()
    )
    print("Dataset built successfully.")

    # --- Step 3: Train BPR Model ---
    # Using the best parameters identified in Phase 1
    print("Training LightFM BPR model with optimal hyperparameters...")
    bpr_model = LightFM(
        loss='bpr',
        no_components=96,
        learning_rate=0.001,
        item_alpha=0.005,
        user_alpha=0.001,
        random_state=42
    )
    
    bpr_model.fit(interactions, epochs=10, num_threads=4) # Using 10 epochs for speed, can be increased to 90 for max performance
    
    print("Model training complete.")

    # --- Step 4: Save Artifacts ---
    print(f"Saving BPR model and dataset mapping to '{ARTIFACTS_DIR}/'...")

    # Save the trained model
    joblib.dump(bpr_model, os.path.join(ARTIFACTS_DIR, 'bpr_model.pkl'))
    
    # CRITICAL: Save the dataset object, as it contains the crucial user/item ID mappings
    joblib.dump(dataset, os.path.join(ARTIFACTS_DIR, 'bpr_dataset_mapping.pkl'))
    
    end_time = time.time()
    print("\n--- Success! BPR Baseline Model Trained ---")
    print(f"Process finished in {end_time - start_time:.2f} seconds.")
    print("The following files have been saved:")
    print(f"- {ARTIFACTS_DIR}/bpr_model.pkl")
    print(f"- {ARTIFACTS_DIR}/bpr_dataset_mapping.pkl")


if __name__ == '__main__':
    train_bpr_model()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import time

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
TRAIN_DATA_FILE = 'train_df.csv'

def train_and_save_model():
    """
    Loads the processed training data, trains the content-based Logistic
    Regression model, and saves the model and vectorizer artifacts.
    """
    print("Starting content-based model training...")
    start_time = time.time()

    # --- Step 1: Load Processed Training Data ---
    train_data_path = os.path.join(PROCESSED_DATA_DIR, TRAIN_DATA_FILE)
    
    try:
        print(f"Loading training data from: {train_data_path}")
        # Note: We're only loading the training data as that's all that's needed for training.
        train_df = pd.read_csv(train_data_path)
        print("Training data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Training data not found at '{train_data_path}'.")
        print("Please run the 'data_loading.py' script first to generate the processed data.")
        return

    # --- Step 2: Feature Engineering and Preparation ---
    print("Preparing content features for training...")
    # Create the 'content' feature from genres and overview
    # We add a placeholder for 'overview' in case the column is missing after loading
    if 'overview' not in train_df.columns:
        train_df['overview'] = ''
        
    train_df['content'] = (train_df['genres'].fillna('') + ' ' + train_df['overview'].fillna(''))
    train_df['content'] = train_df['content'].astype(str) # Ensure content is string
    
    X_train_text = train_df['content']
    y_train = train_df['liked']
    print(f"Prepared {len(X_train_text)} text entries for vectorization.")

    # --- Step 3: Vectorization (TF-IDF with N-grams) ---
    print("Vectorizing text using TF-IDF with N-grams (max_features=5000)...")
    # This matches the best parameters from our notebook experiments [cite: mars-mvp-data-preprocessing-2.ipynb]
    vectorizer_ngram = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf_ngram = vectorizer_ngram.fit_transform(X_train_text)
    print("Vectorization complete.")

    # --- Step 4: Train the Logistic Regression Model ---
    print("Training Logistic Regression model...")
    # Using the best hyperparameters found in Phase 1 [cite: mars-mvp-data-preprocessing-2.ipynb]
    log_reg_ngram = LogisticRegression(C=10, solver='liblinear', random_state=42)
    log_reg_ngram.fit(X_train_tfidf_ngram, y_train)
    print("Model training complete.")

    # --- Step 5: Save the Artifacts ---
    print("Saving model and vectorizer artifacts to the root directory...")
    joblib.dump(log_reg_ngram, 'log_reg_ngram_model.pkl')
    joblib.dump(vectorizer_ngram, 'vectorizer_ngram.pkl')

    end_time = time.time()
    print("\n--- Success! ---")
    print(f"Process finished in {end_time - start_time:.2f} seconds.")
    print("The following files have been saved in your project's root directory:")
    print("- log_reg_ngram_model.pkl")
    print("- vectorizer_ngram.pkl")


if __name__ == '__main__':
    train_and_save_model()
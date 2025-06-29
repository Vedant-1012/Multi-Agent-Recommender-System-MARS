# import pandas as pd
# import numpy as np
# import joblib
# import os
# import time

# try:
#     import faiss
# except ImportError:
#     print("="*70)
#     print("ERROR: The 'faiss-cpu' library is not installed.")
#     print("Please install it by running the following command in your terminal:")
#     print("\npip install faiss-cpu\n")
#     print("="*70)
#     exit()

# # --- Configuration ---
# PROCESSED_DATA_DIR = 'data/processed'
# ARTIFACTS_DIR = './' 
# # --- FIX: Changed the output directory to 'tools' ---
# VECTORSTORE_DIR = 'tools'

# def create_vector_store():
#     """
#     Loads all movie data, creates content vectors using the pre-trained
#     vectorizer, builds a FAISS index for efficient search, and saves the
#     index and movie mapping to the vectorstore directory.
#     """
#     print("Starting Step 1 (Phase 2): Creating the Vector Store...")
#     start_time = time.time()

#     # --- Step 1: Load Processed Data and Artifacts ---
#     print("Loading data and vectorizer...")
#     try:
#         train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_df.csv'))
#         test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_df.csv'))
#         vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer_ngram.pkl'))
#     except FileNotFoundError as e:
#         print(f"ERROR: A required file was not found: {e.filename}")
#         print("Please ensure you have run the setup scripts from Phase 1 first.")
#         return

#     # Combine all movie data and get unique movies
#     all_movies_df = pd.concat([train_df, test_df], ignore_index=True)
#     unique_movies = all_movies_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
#     print(f"Loaded {len(unique_movies)} unique movies.")

#     # --- Step 2: Create Content Embeddings ---
#     print("Creating content features for all unique movies...")
#     if 'overview' not in unique_movies.columns:
#         unique_movies['overview'] = ''
    
#     unique_movies['content'] = (unique_movies['genres'].fillna('') + ' ' + unique_movies['overview'].fillna(''))
    
#     print("Vectorizing movie content using the pre-trained TF-IDF vectorizer...")
#     # This creates the vector representation for every movie
#     movie_vectors = vectorizer.transform(unique_movies['content'].astype(str))
#     # FAISS requires float32 data
#     movie_vectors = movie_vectors.astype(np.float32)

#     # --- Step 3: Build the FAISS Index ---
#     print("Building the FAISS index for similarity search...")
#     num_dimensions = movie_vectors.shape[1]
    
#     # We use IndexFlatIP (Inner Product) because it's equivalent to cosine similarity
#     # for normalized vectors, which is what TfidfVectorizer produces.
#     index = faiss.IndexFlatIP(num_dimensions)
    
#     # Add the movie vectors to the FAISS index
#     index.add(movie_vectors.toarray()) # .toarray() is needed to convert from sparse matrix
    
#     print(f"FAISS index built successfully. Total movies in index: {index.ntotal}")

#     # --- Step 4: Create and Save Artifacts ---
#     print(f"Saving artifacts to '{VECTORSTORE_DIR}/' directory...")
#     # Create the directory if it doesn't exist
#     if not os.path.exists(VECTORSTORE_DIR):
#         os.makedirs(VECTORSTORE_DIR)

#     # Save the FAISS index
#     faiss.write_index(index, os.path.join(VECTORSTORE_DIR, 'movie_index.faiss'))
    
#     # Create a mapping from index position to movie info
#     # This is crucial for retrieving movie details from search results
#     movie_mapping = unique_movies[['movieId', 'title']].to_dict(orient='records')
#     joblib.dump(movie_mapping, os.path.join(VECTORSTORE_DIR, 'movie_mapping.pkl'))

#     end_time = time.time()
#     print("\n--- Success! Vector Store Created ---")
#     print(f"Process finished in {end_time - start_time:.2f} seconds.")
#     print("The following files have been saved:")
#     print(f"- {VECTORSTORE_DIR}/movie_index.faiss")
#     print(f"- {VECTORSTORE_DIR}/movie_mapping.pkl")


# if __name__ == '__main__':
#     create_vector_store()


import pandas as pd
import numpy as np
import joblib
import os
import time

try:
    import faiss
except ImportError:
    print("="*70)
    print("ERROR: The 'faiss-cpu' library is not installed.")
    print("Please install it by running the following command in your terminal:")
    print("\npip install faiss-cpu\n")
    print("="*70)
    exit()

from sentence_transformers import SentenceTransformer

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
VECTORSTORE_DIR = 'tools' 

def create_vector_store():
    """
    Loads movie data, creates semantic vectors using Sentence-BERT, 
    builds a FAISS index, and saves the artifacts.
    """
    print("Starting Step 1: Upgrading to Semantic Vector Store...")
    start_time = time.time()

    # --- Step 1: Load Processed Data ---
    print("Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_df.csv'))
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_df.csv'))
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e.filename}")
        return

    all_movies_df = pd.concat([train_df, test_df], ignore_index=True)
    unique_movies = all_movies_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
    print(f"Loaded {len(unique_movies)} unique movies.")
    
    # --- Step 2: Load Sentence-BERT Model ---
    print("Loading pre-trained Sentence-BERT model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- Step 3: Create Semantic Content Embeddings ---
    print("Creating semantic features for all unique movies...")
    
    # --- FIX: Restore the defensive check for the 'overview' column ---
    if 'overview' not in unique_movies.columns:
        print("WARN: 'overview' column not found. Creating empty column.")
        unique_movies['overview'] = ''
    
    unique_movies['content'] = (unique_movies['genres'].fillna('') + '. ' + unique_movies['overview'].fillna(''))
    
    print("Encoding movie content into semantic vectors... (This may take a few minutes)")
    movie_vectors = model.encode(unique_movies['content'].tolist(), show_progress_bar=True)
    movie_vectors = movie_vectors.astype(np.float32)

    # --- Step 4: Build the FAISS Index ---
    print("Building the new semantic FAISS index...")
    num_dimensions = movie_vectors.shape[1] 
    
    index = faiss.IndexFlatIP(num_dimensions)
    faiss.normalize_L2(movie_vectors)
    index.add(movie_vectors)
    
    print(f"FAISS index built successfully. Total movies in index: {index.ntotal}, Dimensions: {num_dimensions}")

    # --- Step 5: Create and Save Artifacts ---
    print(f"Saving artifacts to '{VECTORSTORE_DIR}/' directory...")
    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)

    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, 'movie_index.faiss'))
    
    movie_mapping = unique_movies[['movieId', 'title']].to_dict(orient='records')
    joblib.dump(movie_mapping, os.path.join(VECTORSTORE_DIR, 'movie_mapping.pkl'))

    end_time = time.time()
    print("\n--- Success! Semantic Vector Store Created ---")
    print(f"Process finished in {end_time - start_time:.2f} seconds.")
    print(f"The following files have been saved to the '{VECTORSTORE_DIR}' directory:")
    print(f"- movie_index.faiss")
    print(f"- movie_mapping.pkl")

if __name__ == '__main__':
    create_vector_store()
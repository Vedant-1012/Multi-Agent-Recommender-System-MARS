# # REASON: This file now uses a "lazy loading" singleton pattern. The heavy
# # MovieRetriever model is not loaded until the `get_movie_retriever`
# # function is called for the first time. This prevents server startup issues.

# import joblib
# import os
# import numpy as np
# import pandas as pd
# import logging

# try:
#     import faiss
# except ImportError:
#     logging.error("ERROR: The 'faiss-cpu' library is not installed.")
#     exit()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- FIX: The global instance is initialized to None ---
# _movie_retriever_instance = None

# class MovieRetriever:
#     """A retriever class that loads the pre-built FAISS index and provides a search interface."""
#     def __init__(self, vectorizer_path: str, index_path: str, mapping_path: str, data_dir: str):
#         logger.info("LAZY LOADING: Initializing MovieRetriever for MARS Project...")
#         self.vectorizer = joblib.load(vectorizer_path)
#         self.index = faiss.read_index(index_path)
#         self.movie_mapping = joblib.load(mapping_path)

#         train_df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
#         test_df = pd.read_csv(os.path.join(data_dir, 'test_df.csv'))
#         all_movies = pd.concat([train_df, test_df], ignore_index=True)
#         self.movie_content_db = all_movies.drop_duplicates(subset=['title']).set_index('title')

#         logger.info("✅ MovieRetriever initialized successfully.")
#         logger.info(f"Loaded FAISS index with {self.index.ntotal} movie vectors.")

#     def search(self, query_text: str, top_k: int = 10) -> list:
#         query_vector = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
#         distances, indices = self.index.search(query_vector, top_k)
#         results = []
#         for i in indices[0]:
#             if i != -1:
#                 movie_info = self.movie_mapping[i]
#                 title = movie_info.get('title')
#                 try:
#                     content = self.movie_content_db.loc[title, 'overview']
#                     plot = str(content) if pd.notna(content) else "No plot overview available."
#                 except KeyError:
#                     plot = "Plot information not found."
#                 results.append({'Title': title, 'Plot': plot})
#         return results

# # --- FIX: A "getter" function to control the lazy loading ---
# def get_movie_retriever():
#     """Initializes and returns the MovieRetriever instance (singleton)."""
#     global _movie_retriever_instance
#     if _movie_retriever_instance is None:
#         _movie_retriever_instance = MovieRetriever(
#             vectorizer_path='vectorizer_ngram.pkl',
#             index_path='vectorstore/movie_index.faiss',
#             mapping_path='vectorstore/movie_mapping.pkl',
#             data_dir='data/processed'
#         )
#     return _movie_retriever_instance



import joblib
import os
import numpy as np
import pandas as pd
import logging

try:
    import faiss
except ImportError:
    logging.error("ERROR: The 'faiss-cpu' library is not installed.")
    exit()

# Import the new, required library
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_movie_retriever_instance = None

class MovieRetriever:
    """A retriever class that loads a pre-built FAISS index and a Sentence-BERT model to provide a semantic search interface."""
    
    # The __init__ signature is updated: it no longer needs a vectorizer_path.
    def __init__(self, index_path: str, mapping_path: str, data_dir: str):
        logger.info("LAZY LOADING: Initializing Semantic MovieRetriever...")
        
        # Load the Sentence-BERT model instead of the TF-IDF vectorizer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.index = faiss.read_index(index_path)
        self.movie_mapping = joblib.load(mapping_path)

        train_df = pd.read_csv(os.path.join(data_dir, 'train_df.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test_df.csv'))
        all_movies = pd.concat([train_df, test_df], ignore_index=True)
        self.movie_content_db = all_movies.drop_duplicates(subset=['title']).set_index('title')

        logger.info("✅ Semantic MovieRetriever initialized successfully.")
        logger.info(f"Loaded FAISS index with {self.index.ntotal} movie vectors.")

    def search(self, query_text: str, top_k: int = 10) -> list:
        """Encodes the query_text using Sentence-BERT and searches the FAISS index."""
        
        # Encode the plain text query into a semantic vector.
        query_vector = self.model.encode([query_text])
        query_vector = query_vector.astype(np.float32)
        # Normalize the query vector to prepare it for cosine similarity search
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in indices[0]:
            if i != -1: # FAISS returns -1 for no result
                movie_info = self.movie_mapping[i]
                title = movie_info.get('title')
                try:
                    content = self.movie_content_db.loc[title, 'overview']
                    plot = str(content) if pd.notna(content) else "No plot overview available."
                except KeyError:
                    plot = "Plot information not found."
                results.append({'Title': title, 'Plot': plot})
        return results

def get_movie_retriever():
    """Initializes and returns the MovieRetriever instance (singleton)."""
    global _movie_retriever_instance
    if _movie_retriever_instance is None:
        # The constructor call is updated, removing the obsolete vectorizer_path.
        _movie_retriever_instance = MovieRetriever(
            index_path='tools/movie_index.faiss',
            mapping_path='tools/movie_mapping.pkl',
            data_dir='data/processed'
        )
    return _movie_retriever_instance
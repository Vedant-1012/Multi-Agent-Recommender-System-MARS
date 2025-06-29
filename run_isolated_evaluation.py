import pandas as pd
import numpy as np
import joblib
import os
import time
from tqdm import tqdm

# We will handle imports carefully to avoid module-level loading
from lightfm import LightFM

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
ARTIFACTS_DIR = 'artifacts'
VECTORSTORE_DIR = 'vectorstore'
TOP_K = 10
SAMPLE_SIZE = 1000

# --- Helper functions ---
# These functions now take the loaded models and data as arguments
# instead of relying on global instances.

def get_bpr_recommendations(user_id, model, dataset, train_interactions, id_to_title_map, k):
    """Generates BPR recommendations using pre-loaded artifacts."""
    try:
        user_internal_id = dataset.mapping()[0][user_id]
        item_internal_ids = np.arange(len(dataset.mapping()[2]))
        scores = model.predict(user_internal_id, item_internal_ids)
        known_positives = train_interactions.getrow(user_internal_id).indices
        scores[known_positives] = -np.inf
        top_item_internal_ids = np.argsort(-scores)[:k]
        
        id_map = dataset.mapping()[2]
        reversed_id_map = {v: k for k, v in id_map.items()}
        
        recommended_ids = [reversed_id_map.get(i) for i in top_item_internal_ids]
        return [id_to_title_map.get(mid) for mid in recommended_ids if mid is not None]
    except (KeyError, IndexError):
        return []

def get_content_recommendations(movie_title, seen_titles, vectorizer, faiss_index, index_to_title_map, k):
    """Generates content-based recommendations using pre-loaded artifacts."""
    # This logic is simplified from the critic agent for direct use here
    if movie_title in _data_manager.cold_start_titles:
        base_query = _data_manager.cold_start_plots.loc[_data_manager.cold_start_plots['title'] == movie_title, 'plot'].iloc[0]
    else:
        try:
            genres = _data_manager.movie_info_db.loc[movie_title, 'genres']
            base_query = f"A movie in the {genres.replace('|', ' ')} genre."
        except KeyError:
            return []

    query_vector = vectorizer.transform([base_query]).toarray().astype(np.float32)
    _, indices = faiss_index.search(query_vector, k + 5)
    
    recommended_titles = [index_to_title_map.get(i) for i in indices[0] if i != -1]
    
    # Filter out movies the user has already seen
    final_recs = [title for title in recommended_titles if title and title not in seen_titles][:k]
    return final_recs

def calculate_metrics(recommendations, ground_truth):
    """Calculates Precision@k and Recall@k."""
    hits = len(set(recommendations).intersection(ground_truth))
    precision = hits / len(recommendations) if recommendations else 0
    recall = hits / len(ground_truth) if ground_truth else 0
    return precision, recall


# --- Main Evaluation Script ---

# This is a small helper class to hold all the data loaded by our agents
class _DataManager:
    def __init__(self):
        self.cold_start_plots = pd.read_csv('generated_plots.csv')
        self.cold_start_titles = set(self.cold_start_plots['title'].unique())
        # The movie_info_db will be loaded in the main function
        self.movie_info_db = None

_data_manager = _DataManager()

def run_efficient_isolated_evaluation():
    print("Starting EFFICIENT Isolated Evaluation: BPR vs. Content-Based...")
    if SAMPLE_SIZE:
        print(f"--- RUNNING ON A SAMPLE OF {SAMPLE_SIZE} USERS ---")
    start_time = time.time()

    # --- Step 1: Load ALL resources ONCE ---
    print("Loading all data and model artifacts once...")
    try:
        # Load core data
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_df.csv'))
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_df.csv'))
        
        # Load BPR artifacts
        bpr_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'bpr_model.pkl'))
        bpr_dataset = joblib.load(os.path.join(ARTIFACTS_DIR, 'bpr_dataset_mapping.pkl'))
        (train_interactions, _) = bpr_dataset.build_interactions(
            (row['userId'], row['movieId']) for _, row in train_df.iterrows()
        )

        # Load Content-based artifacts
        vectorizer = joblib.load('vectorizer_ngram.pkl')
        faiss_index = faiss.read_index(os.path.join(VECTORSTORE_DIR, 'movie_index.faiss'))
        movie_mapping = joblib.load(os.path.join(VECTORSTORE_DIR, 'movie_mapping.pkl'))
        # Create a mapping from faiss index position -> title
        index_to_title_map = {i: item['title'] for i, item in enumerate(movie_mapping)}
        
        # Create a mapping from movieId -> title for BPR
        all_movies = pd.concat([train_df, test_df])
        id_to_title_map = all_movies.drop_duplicates('movieId').set_index('movieId')['title'].to_dict()
        
        # Load data needed by the content-based query generator
        _data_manager.movie_info_db = all_movies.drop_duplicates('title').set_index('title')

    except Exception as e:
        print(f"ERROR: Failed to load a required file. Please check all paths and previous steps. Details: {e}")
        return
    print("✅ All resources loaded successfully.")

    # --- Step 2: Run Evaluation Loop ---
    bpr_metrics, content_metrics = [], []
    all_test_users = test_df['userId'].unique()
    # --- THIS IS THE FIX ---
    users_to_evaluate = np.random.choice(all_test_users, SAMPLE_SIZE, replace=False) if SAMPLE_SIZE else all_test_users
    # --- END FIX ---
        
    test_user_groups = test_df[test_df['userId'].isin(users_to_evaluate)].groupby('userId')
    train_history_groups = train_df.groupby('userId')['title'].apply(list).to_dict()

    print(f"Evaluating {len(test_user_groups)} users...")
    for user_id, user_data in tqdm(test_user_groups, desc="Evaluating Users"):
        ground_truth_titles = set(user_data[user_data['liked'] == 1]['title'])
        if not ground_truth_titles: continue
        
        # --- Strategy 1: BPR Recommendations ---
        bpr_recs = get_bpr_recommendations(user_id, bpr_model, bpr_dataset, train_interactions, id_to_title_map, TOP_K)
        
        # --- Strategy 2: Content-Based Recommendations ---
        user_train_history = train_history_groups.get(user_id, [])
        last_liked_movie_title = user_data.sort_values('timestamp', ascending=False).iloc[0]['title']
        seen_titles = set(user_train_history).union({last_liked_movie_title})
        
        content_recs = get_content_recommendations(last_liked_movie_title, seen_titles, vectorizer, faiss_index, index_to_title_map, TOP_K)

        # --- Calculate metrics ---
        bpr_precision, bpr_recall = calculate_metrics(bpr_recs, ground_truth_titles)
        content_precision, content_recall = calculate_metrics(content_recs, ground_truth_titles)
        
        bpr_metrics.append({'precision': bpr_precision, 'recall': bpr_recall})
        content_metrics.append({'precision': content_precision, 'recall': content_recall})

    # --- Step 3: Display Results ---
    bpr_results = pd.DataFrame(bpr_metrics).mean()
    content_results = pd.DataFrame(content_metrics).mean()

    print("\n--- Isolated Strategy Performance ---")
    print(f"{'Metric':<15} | {'BPR (Personalization)':<25} | {'Content-Based (Cold-Start)':<25}")
    print("-" * 75)
    print(f"{f'Precision@{TOP_K}':<15} | {bpr_results['precision']:<25.4f} | {content_results['precision']:<25.4f}")
    print(f"{f'Recall@{TOP_K}':<15} | {bpr_results['recall']:<25.4f} | {content_results['recall']:<25.4f}")
    print("-" * 75)
    
    end_time = time.time()
    print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    # Add a check for faiss library at the top-level
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss-cpu is not installed. Please run `pip install faiss-cpu`")
    else:
        run_efficient_isolated_evaluation()

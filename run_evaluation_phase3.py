import pandas as pd
import numpy as np
import joblib
import os
import time
from tqdm import tqdm
from itertools import combinations

# --- Import our project modules ---
from manager_agent.sub_agents.recommender_agent.agent import generate_search_query
from storage.vector_db import get_movie_retriever 
from tools.personalization_tools import get_personalized_recommendations

# --- Import new libraries for diversity calculation ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
ARTIFACTS_DIR = 'artifacts'
TOP_K = 10
COLD_START_THRESHOLD = 5 
SAMPLE_SIZE = 1000 

def get_content_based_recommendations(last_liked_movie_title, user_liked_history_titles, k, movie_retriever):
    """Simulates the content-based RecommenderAgent for cold-start users."""
    base_query = generate_search_query(last_liked_movie_title)
    search_results = movie_retriever.search(base_query, top_k=k + 20) 
    
    recommended_titles = []
    if search_results:
        seen_movies = set(user_liked_history_titles)
        seen_movies.add(last_liked_movie_title)
        for rec in search_results:
            title = rec.get('Title')
            if title and title not in seen_movies:
                recommended_titles.append(title)
    return recommended_titles[:k]

def calculate_accuracy_metrics(recommendations, ground_truth):
    """Calculates Precision@k and Recall@k."""
    hits = len(set(recommendations).intersection(set(ground_truth)))
    precision = hits / len(recommendations) if recommendations else 0
    recall = hits / len(ground_truth) if ground_truth else 0
    return precision, recall

def calculate_diversity(recommendations: list, vectors_map: dict):
    """Calculates Intra-List Diversity (ILD) using cosine distance."""
    if not recommendations or len(recommendations) < 2:
        return 0.0

    rec_vectors = [vectors_map[title] for title in recommendations if title in vectors_map]
    if len(rec_vectors) < 2:
        return 0.0

    # Calculate cosine similarity for all pairs of items in the list
    similarity_matrix = cosine_similarity(rec_vectors)
    
    # We only need the upper triangle of the matrix, excluding the diagonal
    upper_triangle_indices = np.triu_indices(len(rec_vectors), k=1)
    pairwise_similarities = similarity_matrix[upper_triangle_indices]
    
    if len(pairwise_similarities) == 0:
        return 0.0
        
    # Diversity = 1 - Average Similarity
    avg_similarity = np.mean(pairwise_similarities)
    return 1 - avg_similarity

# --- Main Evaluation Script ---
def run_final_evaluation():
    """
    Main function to run the definitive head-to-head evaluation, now including a diversity metric.
    """
    print("Starting Definitive Evaluation: Segmented Analysis with Diversity...")
    if SAMPLE_SIZE:
        print(f"--- RUNNING ON A SAMPLE OF {SAMPLE_SIZE} USERS ---")
    start_time = time.time()

    # --- Step 1: Load all necessary data and models ---
    print("Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_df.csv'))
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_df.csv'))
    except FileNotFoundError as e:
        print(f"ERROR: A required data file was not found: {e.filename}")
        return
    
    print("Initializing models (lazy loading)...")
    movie_retriever = get_movie_retriever()
    
    # --- NEW: Pre-calculate all movie vectors for diversity calculation ---
    print("Pre-calculating all movie embeddings for diversity metric...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    all_movies_df = pd.concat([train_df, test_df]).drop_duplicates(subset=['title']).reset_index()
    if 'overview' not in all_movies_df.columns:
        all_movies_df['overview'] = ''
    all_movies_df['content'] = all_movies_df['genres'].fillna('') + '. ' + all_movies_df['overview'].fillna('')
    titles = all_movies_df['title'].tolist()
    vectors = sbert_model.encode(all_movies_df['content'].tolist(), show_progress_bar=True)
    title_to_vector_map = {title: vector for title, vector in zip(titles, vectors)}
    print("Embeddings calculated.")
    
    print("Data and models ready.")

    # --- Step 2: Run Evaluation Loop ---
    warm_start_metrics = []
    cold_start_metrics = []
    
    all_test_users = test_df['userId'].unique()
    if SAMPLE_SIZE and SAMPLE_SIZE < len(all_test_users):
        users_to_evaluate = np.random.choice(all_test_users, SAMPLE_SIZE, replace=False)
    else:
        users_to_evaluate = all_test_users
        
    test_user_groups = test_df[test_df['userId'].isin(users_to_evaluate)].groupby('userId')
    train_history_groups = train_df.groupby('userId')['title'].apply(list).to_dict()

    print(f"Evaluating {len(test_user_groups)} users...")
    for user_id, user_data in tqdm(test_user_groups, desc="Evaluating Users"):
        user_id_str = str(user_id)
        ground_truth_titles = set(user_data[user_data['liked'] == 1]['title'])
        if not ground_truth_titles:
            continue
            
        bpr_rec_titles_result = get_personalized_recommendations(user_id_str)
        if bpr_rec_titles_result["recommendations"] and "Since you're a new user" in bpr_rec_titles_result["recommendations"][0]:
            bpr_rec_titles = bpr_rec_titles_result["recommendations"][1:] 
        else:
            bpr_rec_titles = bpr_rec_titles_result.get("recommendations", [])

        user_train_history = train_history_groups.get(user_id, [])
        last_liked_movie_title_series = user_data.sort_values('timestamp', ascending=False)['title']
        if last_liked_movie_title_series.empty:
            continue
        last_liked_movie_title = last_liked_movie_title_series.iloc[0]

        is_cold_start = len(user_train_history) < COLD_START_THRESHOLD
        
        if is_cold_start:
            mars_rec_titles = get_content_based_recommendations(last_liked_movie_title, user_train_history, TOP_K, movie_retriever)
        else:
            mars_rec_titles = bpr_rec_titles

        bpr_precision, bpr_recall = calculate_accuracy_metrics(bpr_rec_titles, ground_truth_titles)
        mars_precision, mars_recall = calculate_accuracy_metrics(mars_rec_titles, ground_truth_titles)
        
        # --- NEW: Calculate diversity for both recommendation lists ---
        bpr_diversity = calculate_diversity(bpr_rec_titles, title_to_vector_map)
        mars_diversity = calculate_diversity(mars_rec_titles, title_to_vector_map)
        
        metrics_to_add = {
            'bpr_precision': bpr_precision, 'bpr_recall': bpr_recall, 'bpr_diversity': bpr_diversity,
            'mars_precision': mars_precision, 'mars_recall': mars_recall, 'mars_diversity': mars_diversity
        }

        if is_cold_start:
            cold_start_metrics.append(metrics_to_add)
        else:
            warm_start_metrics.append(metrics_to_add)

    # --- Step 3: Aggregate and Display Segmented Results ---
    print("\n--- Definitive Evaluation Results: Segmented Analysis ---")
    
    if warm_start_metrics:
        warm_results_df = pd.DataFrame(warm_start_metrics)
        print(f"\n--- WARM-START Users ({len(warm_results_df)} users) ---")
        print(f"{'Metric':<15} | {'BPR Baseline':<15} | {'MARS v2 System':<20}")
        print("-" * 60)
        print(f"{f'Precision@{TOP_K}':<15} | {warm_results_df['bpr_precision'].mean():<15.4f} | {warm_results_df['mars_precision'].mean():<20.4f}")
        print(f"{f'Recall@{TOP_K}':<15} | {warm_results_df['bpr_recall'].mean():<15.4f} | {warm_results_df['mars_recall'].mean():<20.4f}")
        print(f"{f'Diversity@{TOP_K}':<15} | {warm_results_df['bpr_diversity'].mean():<15.4f} | {warm_results_df['mars_diversity'].mean():<20.4f}")
    else:
        print("\n--- No WARM-START Users in this sample ---")

    if cold_start_metrics:
        cold_results_df = pd.DataFrame(cold_start_metrics)
        print(f"\n--- COLD-START Users ({len(cold_results_df)} users) ---")
        print(f"{'Metric':<15} | {'BPR Baseline':<15} | {'MARS v2 System':<20}")
        print("-" * 60)
        print(f"{f'Precision@{TOP_K}':<15} | {cold_results_df['bpr_precision'].mean():<15.4f} | {cold_results_df['mars_precision'].mean():<20.4f}")
        print(f"{f'Recall@{TOP_K}':<15} | {cold_results_df['bpr_recall'].mean():<15.4f} | {cold_results_df['mars_recall'].mean():<20.4f}")
        print(f"{f'Diversity@{TOP_K}':<15} | {cold_results_df['bpr_diversity'].mean():<15.4f} | {cold_results_df['mars_diversity'].mean():<20.4f}")
    else:
        print("\n--- No COLD-START Users in this sample ---")

    print("-" * 60)
    end_time = time.time()
    print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    run_final_evaluation()
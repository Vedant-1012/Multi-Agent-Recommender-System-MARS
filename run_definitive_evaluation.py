import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
import os

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
ARTIFACTS_DIR = './' 

# --- Mock objects to simulate your system's components ---
class MockMovieRetriever:
    """A mock of your vector DB retriever to simulate search."""
    def __init__(self, vectorizer, data_df):
        self.vectorizer = vectorizer
        # Make a copy to prevent modifying the original dataframe
        self.data_df = data_df.drop_duplicates(subset=['title']).reset_index(drop=True).copy()
        
        if 'overview' not in self.data_df.columns:
            self.data_df['overview'] = ''
        
        self.data_df['content'] = (self.data_df['genres'].fillna('') + ' ' + self.data_df['overview'].fillna(''))
        self.content_matrix = self.vectorizer.transform(self.data_df['content'].astype(str))

    def search(self, query_text, top_k=10):
        query_vector = self.vectorizer.transform([query_text])
        similarities = (self.content_matrix * query_vector.T).toarray().flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [{'Title': self.data_df.iloc[i]['title'], 'Plot': self.data_df.iloc[i]['content']} for i in top_indices]
        return results

def mock_recommend_movies(base_query, retriever_instance):
    """A mock of your recommend_movies tool."""
    recommendations = retriever_instance.search(base_query, top_k=5)
    return {"recommendations": recommendations}

# --- Main Evaluation Script ---
def run_evaluation():
    print("Starting Final End-to-End Evaluation for Phase 1 (Corrected)...")

    # --- Step 1: Load All Artifacts ---
    print("Loading all data and models...")
    try:
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_df.csv'))
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_df.csv'))
        df_for_retriever = pd.concat([train_df, test_df], ignore_index=True)
        generated_plots_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'generated_plots.csv'))
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer_ngram.pkl'))
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e.filename}")
        return

    # --- Step 2: Initialize the BASE retriever ---
    base_movie_retriever = MockMovieRetriever(vectorizer, df_for_retriever)
    print("Artifacts loaded and base retriever initialized.")

    # --- Step 3: Run the Evaluation Loop ---
    results_data = []
    cold_start_titles = generated_plots_df['title'].unique()
    ground_truth_df = test_df[test_df['title'].isin(cold_start_titles)]
    users_to_test = ground_truth_df['userId'].unique()

    print(f"\nFound {len(cold_start_titles)} cold-start movies and {len(users_to_test)} users who rated them in the test set.")
    print("Evaluating performance...")

    for user_id in tqdm(users_to_test, desc="Evaluating Users"):
        actual_liked_movies = set(ground_truth_df[(ground_truth_df['userId'] == user_id) & (ground_truth_df['liked'] == 1)]['title'])
        if not actual_liked_movies:
            continue

        for title in actual_liked_movies:
            # --- Scenario A (Original Data) ---
            original_genre = df_for_retriever[df_for_retriever['title'] == title]['genres'].iloc[0]
            base_query_original = f"A movie in the {original_genre} genre."
            recommendations_original = mock_recommend_movies(base_query_original, base_movie_retriever)
            recommended_titles_original = {rec['Title'] for rec in recommendations_original['recommendations']}

            # --- Scenario B (LLM Data) ---
            llm_plot = generated_plots_df[generated_plots_df['title'] == title]['plot'].iloc[0]
            
            # --- THIS IS THE FIX ---
            # Create a TEMPORARY, UPDATED retriever where the movie's content IS the LLM plot
            llm_retriever = MockMovieRetriever(vectorizer, df_for_retriever)
            movie_index = llm_retriever.data_df.index[llm_retriever.data_df['title'] == title].tolist()[0]
            llm_retriever.data_df.loc[movie_index, 'content'] = llm_plot
            # Re-vectorize the content matrix with the updated plot
            llm_retriever.content_matrix = vectorizer.transform(llm_retriever.data_df['content'].astype(str))
            # --- END FIX ---

            base_query_llm = llm_plot # The query is the plot itself
            recommendations_llm = mock_recommend_movies(base_query_llm, llm_retriever)
            recommended_titles_llm = {rec['Title'] for rec in recommendations_llm['recommendations']}
            
            # --- Calculate Metrics ---
            hits_original = len(recommended_titles_original.intersection(actual_liked_movies))
            hits_llm = len(recommended_titles_llm.intersection(actual_liked_movies))
            
            precision_original = hits_original / 5.0
            recall_original = hits_original / len(actual_liked_movies)
            
            precision_llm = hits_llm / 5.0
            recall_llm = hits_llm / len(actual_liked_movies)
            
            results_data.append([precision_original, recall_original, precision_llm, recall_llm])

    # --- Step 4: Aggregate and Display Final Results ---
    results_df = pd.DataFrame(results_data, columns=['precision_orig', 'recall_orig', 'precision_llm', 'recall_llm'])

    final_metrics = {
        "Precision_Original": results_df['precision_orig'].mean(),
        "Recall_Original": results_df['recall_orig'].mean(),
        "Precision_LLM": results_df['precision_llm'].mean(),
        "Recall_LLM": results_df['recall_llm'].mean()
    }

    print("\n--- Final Cold-Start Performance: LLM-Generated vs. Original Data ---")
    print(f"{'Metric':<12} | {'Original Data':<15} | {'LLM-Generated Data':<20} | {'Improvement'}")
    print("-" * 70)

    precision_improvement = final_metrics['Precision_LLM'] - final_metrics['Precision_Original']
    recall_improvement = final_metrics['Recall_LLM'] - final_metrics['Recall_Original']

    print(f"{'Precision@5':<12} | {final_metrics['Precision_Original']:<15.4f} | {final_metrics['Precision_LLM']:<20.4f} | {precision_improvement:+.4f}")
    print(f"{'Recall@5':<12} | {final_metrics['Recall_Original']:<15.4f} | {final_metrics['Recall_LLM']:<20.4f} | {recall_improvement:+.4f}")
    print("-" * 70)

if __name__ == '__main__':
    run_evaluation()
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import os
import time

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
TEST_DATA_FILE = 'test_df.csv'
GENERATED_PLOTS_FILE = 'generated_plots.csv'
MODEL_FILE = 'log_reg_ngram_model.pkl'
VECTORIZER_FILE = 'vectorizer_ngram.pkl'

def run_final_evaluation():
    """
    Loads all necessary data and model artifacts to perform the final
    head-to-head evaluation for the cold-start movie experiment.
    """
    print("Starting Final Evaluation Script for Phase 1...")
    start_time = time.time()

    # --- Step 1: Load All Necessary Artifacts ---
    print("Loading all data and pre-trained models...")
    
    try:
        # Load the test dataframe that was created by data_loading.py
        test_df_path = os.path.join(PROCESSED_DATA_DIR, TEST_DATA_FILE)
        test_df = pd.read_csv(test_df_path)

        # Load the LLM-generated plots
        generated_plots_df = pd.read_csv(GENERATED_PLOTS_FILE)

        # Load our best content-based model and its vectorizer
        log_reg_model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)

        print("All artifacts loaded successfully.")

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found: {e.filename}")
        print("Please ensure you have run the three setup scripts first:")
        print("1. python data_loading.py")
        print("2. python train_content_model.py")
        print("3. python generate_plots.py")
        return

    # --- Step 2: Prepare the Evaluation Data ---
    # Isolate the cold-start movies that are present in our test set
    cold_start_titles = generated_plots_df['title'].unique()
    evaluation_df = test_df[test_df['title'].isin(cold_start_titles)].copy()

    # Add the generated plots to this evaluation dataframe
    evaluation_df = pd.merge(evaluation_df, generated_plots_df, on='title', how='left')
    
    # This is the fix from the notebook to correct the column name mismatch
    #
    evaluation_df.rename(columns={'plot': 'generated_plot'}, inplace=True)

    # The ground truth is the 'liked' column from the actual user ratings
    y_true = evaluation_df['liked']

    # --- Step 3: Run Scenario A (Control Group - Using Original Content) ---
    print("\nRunning Scenario A: Evaluating with ORIGINAL content (genres only)...")
    X_test_original = vectorizer.transform(evaluation_df['genres'].astype(str))
    y_pred_original = log_reg_model.predict(X_test_original)
    y_proba_original = log_reg_model.predict_proba(X_test_original)[:, 1]
    
    precision_original = precision_score(y_true, y_pred_original, zero_division=0)
    recall_original = recall_score(y_true, y_pred_original, zero_division=0)
    roc_auc_original = roc_auc_score(y_true, y_proba_original)

    # --- Step 4: Run Scenario B (Experimental Group - Using LLM Content) ---
    print("Running Scenario B: Evaluating with LLM-GENERATED content...")
    X_test_llm = vectorizer.transform(evaluation_df['generated_plot'].astype(str))
    y_pred_llm = log_reg_model.predict(X_test_llm)
    y_proba_llm = log_reg_model.predict_proba(X_test_llm)[:, 1]

    precision_llm = precision_score(y_true, y_pred_llm, zero_division=0)
    recall_llm = recall_score(y_true, y_pred_llm, zero_division=0)
    roc_auc_llm = roc_auc_score(y_true, y_proba_llm)

    # --- Step 5: Display Final Comparison ---
    print("\n--- Cold-Start Performance: LLM-Generated vs. Original Data ---")
    print(f"{'Metric':<12} | {'Original Data':<15} | {'LLM-Generated Data':<20} | {'Improvement'}")
    print("-" * 70)
    
    precision_improvement = precision_llm - precision_original
    recall_improvement = recall_llm - recall_original
    roc_auc_improvement = roc_auc_llm - roc_auc_original
    
    print(f"{'Precision':<12} | {precision_original:<15.4f} | {precision_llm:<20.4f} | {precision_improvement:+.4f}")
    print(f"{'Recall':<12} | {recall_original:<15.4f} | {recall_llm:<20.4f} | {recall_improvement:+.4f}")
    print(f"{'ROC-AUC':<12} | {roc_auc_original:<15.4f} | {roc_auc_llm:<20.4f} | {roc_auc_improvement:+.4f}")
    print("-" * 70)
    
    end_time = time.time()
    print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    run_final_evaluation()
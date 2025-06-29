# tools/personalization_tools.py

import joblib
import os
import numpy as np
import pandas as pd
import logging
from typing import Union

try:
    from lightfm import LightFM
except ImportError:
    logging.error("LightFM library not found.")

ARTIFACTS_DIR = 'artifacts'
PROCESSED_DATA_DIR = 'data/processed'

_bpr_recommender_instance = None

class BPRRecommender:
    # ... (class content remains exactly the same) ...
    def __init__(self):
        logging.info("LAZY LOADING: Initializing BPR Recommender Tool...")
        model_path = os.path.join(ARTIFACTS_DIR, 'bpr_model.pkl')
        dataset_path = os.path.join(ARTIFACTS_DIR, 'bpr_dataset_mapping.pkl')
        
        self.model = joblib.load(model_path)
        self.dataset = joblib.load(dataset_path)
        
        self.train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_df.csv'))
        self.test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_df.csv'))
        
        movie_titles = pd.concat([self.train_df, self.test_df])[['movieId', 'title']].drop_duplicates()
        self.movie_id_to_title = movie_titles.set_index('movieId')['title'].to_dict()
        (self.train_interactions, _) = self.dataset.build_interactions(
            (row['userId'], row['movieId']) for _, row in self.train_df.iterrows()
        )
        logging.info("✅ BPR Recommender Tool initialized successfully.")

    def get_recommendations(self, user_id: int, k: int = 10) -> list[str]:
        user_internal_id = self.dataset.mapping()[0][user_id]
        item_internal_ids = np.arange(len(self.dataset.mapping()[2]))
        
        scores = self.model.predict(user_internal_id, item_internal_ids)
        known_positives = self.train_interactions.getrow(user_internal_id).indices
        scores[known_positives] = -np.inf
        
        top_item_internal_ids = np.argsort(-scores)[:k]
        id_map = self.dataset.mapping()[2]
        reversed_id_map = {v: k for k, v in id_map.items()}
        
        recommended_movie_ids = [reversed_id_map[i] for i in top_item_internal_ids]
        return [self.movie_id_to_title.get(mid, "Unknown Movie") for mid in recommended_movie_ids]

    def get_most_popular(self, k: int = 10) -> list[str]:
        logging.info(f"BPR model fallback: getting top {k} most popular movies.")
        all_ratings = pd.concat([self.train_df, self.test_df])
        movie_counts = all_ratings.groupby('title').size()
        top_movies = movie_counts.sort_values(ascending=False).head(k).index.tolist()
        return top_movies

def get_bpr_recommender():
    # ... (function remains the same) ...
    global _bpr_recommender_instance
    if _bpr_recommender_instance is None:
        _bpr_recommender_instance = BPRRecommender()
    return _bpr_recommender_instance

def get_personalized_recommendations(user_id: str) -> dict:
    # ... (function remains the same) ...
    logging.info(f"TOOL EXECUTED: get_personalized_recommendations(user_id={user_id})")
    bpr_recommender = get_bpr_recommender()
    try:
        int_user_id = int(user_id)
        recommendations = bpr_recommender.get_recommendations(int_user_id)
        return {"recommendations": recommendations}
    except (ValueError, TypeError, KeyError):
        logging.warning(f"User ID '{user_id}' not found in BPR model. Falling back to most popular.")
        recommendations = bpr_recommender.get_most_popular(k=10)
        recommendations.insert(0, "Since you're a new user, here are some of the most popular movies:")
        return {"recommendations": recommendations}

# --- NEW FUNCTION ---
# This new tool gives the Manager Agent direct access to the most popular list.
def get_top_popular_movies(top_k: int = 10) -> dict:
    """Gets a list of the globally most popular movies."""
    logging.info(f"TOOL EXECUTED: get_top_popular_movies(top_k={top_k})")
    recommender = get_bpr_recommender()
    popular_movies = recommender.get_most_popular(k=top_k)
    return {"recommendations": popular_movies}
import pandas as pd
import logging
import os

# --- UPGRADE: Point to our new, processed MovieLens data ---
PROCESSED_DATA_DIR = 'data/processed'
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_df.csv')
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_df.csv')

MOVIE_DATA = pd.DataFrame()
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    # Combine train and test sets to have a complete view of all movies
    MOVIE_DATA = pd.concat([train_df, test_df], ignore_index=True)
    logging.info("✅ MovieLens 20M data loaded successfully for data access.")
except Exception as e:
    logging.error(f"❌ Failed to load MovieLens data from '{PROCESSED_DATA_DIR}': {e}")


def search_movies_by_keywords(query: str, top_k: int = 5) -> list:
    """Keyword-based search across Title and Genres."""
    if MOVIE_DATA.empty:
        logging.warning("⚠️ No movie data available for keyword search.")
        return []

    query_lower = query.lower()
    
    # Search on a de-duplicated list of movies to avoid multiple entries for the same film
    unique_movies = MOVIE_DATA.drop_duplicates(subset=['title'])
    
    matches = unique_movies[
        unique_movies['title'].str.strip().str.lower().str.contains(query_lower, na=False) |
        unique_movies['genres'].str.strip().str.lower().str.contains(query_lower, na=False)
    ]
    results = []
    for _, row in matches.head(top_k).iterrows():
        results.append({
            "title": row["title"],
            "plot": row.get("overview", "No plot available."),
            "genres": row.get("genres", "N/A")
        })
    return results


def get_rating_by_title(title: str) -> dict:
    """
    Retrieve the AVERAGE rating for a movie title from the MovieLens dataset.
    """
    if MOVIE_DATA.empty:
        return None

    title_lower = title.lower()
    
    # Find all rating instances for the specified movie
    matches = MOVIE_DATA[MOVIE_DATA['title'].str.strip().str.lower() == title_lower]
    
    if matches.empty:
        return {"rating": "Not found in dataset."}

    # --- THIS IS THE FIX ---
    # Calculate the mean of the 'rating' column for all matching rows
    avg_rating = matches['rating'].mean()
    # --- END FIX ---
    
    # Return the calculated average, formatted to two decimal places
    return { "rating": f"{avg_rating:.2f}" }


# import pandas as pd
# import os
# import re
# import logging
# from google.adk.agents import Agent
# from tools.movie_tools import recommend_movies # We still need this tool

# # ### LOGIC MOVED FROM CRITIC_AGENT ###
# class _CriticDataManager:
#     def __init__(self):
#         self.title_to_genres_map: dict = {}
#         self.lowercase_to_original_title_map: dict = {}
#         self.cold_start_plots_map: dict = {}
#         self._initialize_data()

#     def _initialize_data(self):
#         print("Initializing Recommender Agent's internal data manager...")
#         try:
#             train_df = pd.read_csv('data/processed/train_df.csv')
#             test_df = pd.read_csv('data/processed/test_df.csv')
#             all_movies = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['title'])
#             all_movies['title_without_year'] = all_movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
#             for _, row in all_movies.iterrows():
#                 lookup_key = row['title_without_year'].strip().lower()
#                 self.title_to_genres_map[lookup_key] = row['genres']
#             cold_start_plots_df = pd.read_csv('generated_plots.csv')
#             cold_start_plots_df['title_without_year'] = cold_start_plots_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
#             for _, row in cold_start_plots_df.iterrows():
#                 lookup_key_cs = row['title_without_year'].strip().lower()
#                 self.cold_start_plots_map[lookup_key_cs] = row['plot']
#             print("✅ Recommender Agent's data manager initialized successfully.")
#         except Exception as e:
#             print(f"❌ FATAL ERROR [RecommenderAgent]: Failed to initialize data - {e}")

# _data_manager = _CriticDataManager()

# # --- FIX: This function is now simpler. It ONLY returns the search query string. ---
# def generate_search_query(movie_title: str) -> str:
#     """
#     Generates a descriptive search query string for a given movie title.
#     """
#     logging.info(f"TOOL EXECUTED [RecommenderAgent]: generate_search_query(movie_title='{movie_title}')")
#     processed_movie_title = movie_title.strip().lower()
#     processed_movie_title = re.sub(r'\s*\(\d{4}\)\s*$', '', processed_movie_title)

#     if processed_movie_title in _data_manager.cold_start_plots_map:
#         return _data_manager.cold_start_plots_map[processed_movie_title]
#     elif processed_movie_title in _data_manager.title_to_genres_map:
#         genres = _data_manager.title_to_genres_map[processed_movie_title]
#         return f"A movie in the {genres.replace('|', ' ')} genre, similar to {movie_title}."
#     else:
#         return f"A movie with themes similar to '{movie_title}'."

# # ### NEW RECOMMENDER AGENT DEFINITION ###
# recommender_agent = Agent(
#     name="recommender_agent",
#     description="Recommends movies for new users by generating a semantic query and filtering based on the user's profile.",
#     # --- NEW, FINAL INSTRUCTION ---
#     instruction="""
#     You are a cold-start movie recommendation specialist. You have been given a user's `request` and their full profile, `{user_profile}`.

#     Your job is a three-step process:

#     **STEP 1: IDENTIFY THE SEED MOVIE.**
#     - From the user's `request`, identify the movie title they want recommendations for.

#     **STEP 2: GENERATE A SEARCH QUERY.**
#     - You MUST call the `generate_search_query` tool. Pass the extracted seed movie title to its `movie_title` parameter.

#     **STEP 3: GET FINAL RECOMMENDATIONS.**
#     - The `generate_search_query` tool will return a simple search query string. You MUST now call the `recommend_movies` tool.
#     - You MUST pass the following parameters:
#         1. `search_text`: The simple search query string from the previous step.
#         2. `liked_movies`: The list of liked movies from `{user_profile}`.
#         3. `disliked_movies`: The list of disliked movies from `{user_profile}`.

#     **FINALLY: PRESENT THE RESULTS.**
#     - Present the final list of movies from the `recommend_movies` tool to the user.
#     """,
#     # --- UPDATED TOOL LIST ---
#     tools=[generate_search_query, recommend_movies],
#     model="gemini-1.5-pro"
# )



# /manager_agent/sub_agents/recommender_agent/agent.py
# manager_agent/sub_agents/recommender_agent/agent.py

import pandas as pd
import os
import re
import logging
from google.adk.agents import Agent

# This agent no longer calls recommend_movies, so the import is removed.

class _RecommenderDataManager:
    def __init__(self):
        self.cold_start_plots_map: dict = {}
        self.movie_content_db: dict = {}
        self._initialize_data()

    def _initialize_data(self):
        print("Initializing Query Generator (Recommender Agent)...")
        try:
            train_df = pd.read_csv('data/processed/train_df.csv')
            test_df = pd.read_csv('data/processed/test_df.csv')
            all_movies = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset=['title'])
            
            for _, row in all_movies.iterrows():
                title_key = row['title'].strip().lower()
                title_key = re.sub(r'\s*\(\d{4}\)\s*$', '', title_key)
                content = f"{row.get('genres', '')}. {row.get('overview', '')}"
                self.movie_content_db[title_key] = content

            cold_start_plots_df = pd.read_csv('generated_plots.csv')
            for _, row in cold_start_plots_df.iterrows():
                title_key = row['title'].strip().lower()
                title_key = re.sub(r'\s*\(\d{4}\)\s*$', '', title_key)
                self.cold_start_plots_map[title_key] = row['plot']

            print("✅ Query Generator (Recommender Agent) data initialized.")
        except Exception as e:
            print(f"❌ FATAL ERROR [RecommenderAgent]: Failed to initialize data - {e}")

_data_manager = _RecommenderDataManager()

def generate_search_query(movie_title: str) -> str:
    """
    Generates the best possible search query string for a given movie title.
    """
    logging.info(f"TOOL EXECUTED [Query Generator]: generate_search_query(movie_title='{movie_title}')")
    processed_title = movie_title.strip().lower()
    processed_title = re.sub(r'\s*\(\d{4}\)\s*$', '', processed_title)

    if processed_title in _data_manager.cold_start_plots_map:
        return _data_manager.cold_start_plots_map[processed_title]
    
    if processed_title in _data_manager.movie_content_db:
        return _data_manager.movie_content_db[processed_title]

    return f"A movie with themes similar to '{movie_title}'."

# This agent's ONLY job is to call the function above.
recommender_agent = Agent(
    name="query_generator_agent",
    description="Takes a movie title and generates the best possible semantic search query for it.",
    instruction="""
    You are an expert at generating search queries.
    - You will be given a movie title in the user's request.
    - You MUST call the `generate_search_query` tool, passing the movie title to the `movie_title` parameter.
    - You MUST return ONLY the raw string output from the tool.
    """,
    tools=[generate_search_query],
    model="gemini-1.5-pro"
)
# tools/movie_tools.py

import logging
from typing import Optional, List
from storage.vector_db import get_movie_retriever
from storage.movie_data_access import get_rating_by_title, search_movies_by_keywords
from storage.state_schema import UserProfile
from google.adk.tools.tool_context import ToolContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - LOG - %(message)s')

def get_movie_rating(title: str) -> dict:
    # ... (this function remains the same)
    logging.info(f"TOOL EXECUTED: get_movie_rating(title='{title}')")
    rating_info = get_rating_by_title(title)
    if rating_info:
        return {"title": title, "rating": rating_info.get("rating"), "votes": rating_info.get("votes")}
    return {"title": title, "rating": "Not Found", "votes": "N/A"}


def search_movies(query: str) -> dict:
    # ... (this function remains the same)
    logging.info(f"TOOL EXECUTED: search_movies(query='{query}')")
    return {"results": search_movies_by_keywords(query, top_k=5)}


# --- FIX: This function now has a simpler signature and logic ---
def recommend_movies(
    search_text: str, # Changed from critic_output to the simpler search_text
    liked_movies: Optional[List[str]] = None,
    disliked_movies: Optional[List[str]] = None
) -> dict:
    """
    Recommends movies using a semantic search_text and then filters the results.
    """
    logging.info(f"TOOL EXECUTED: recommend_movies with search_text: '{search_text}'")

    movie_retriever = get_movie_retriever()

    # No longer need to parse a complex string, we can use the search_text directly
    recommendations = movie_retriever.search(search_text, top_k=15)
    logging.info(f"Vector search returned {len(recommendations)} raw recommendations.")

    final_recommendations = []
    seen_titles = set()
    if liked_movies:
        seen_titles.update(liked_movies)
    if disliked_movies:
        seen_titles.update(disliked_movies)

    if recommendations:
        for rec in recommendations:
            title = rec.get('Title')
            if title and title not in seen_titles:
                final_recommendations.append(rec)

    logging.info(f"Filtered recommendations: {len(final_recommendations)} movies.")
    return {"recommendations": final_recommendations[:5]}


# --- FIX: Replace the empty function with this complete version ---
def update_user_preferences(
    tool_context: ToolContext,
    liked_movie: Optional[str] = None,
    disliked_movie: Optional[str] = None
) -> dict:
    """Saves a user's movie preference by safely updating the session state."""
    logging.info(f"TOOL EXECUTED: update_user_preferences(liked='{liked_movie}', disliked='{disliked_movie}')")

    # Get the current user profile from the session state
    profile_dict = tool_context.state.get("user_profile")
    if not profile_dict:
        return {"status": "error", "message": "User profile not found in session."}

    # Load it into the UserProfile data model
    user_profile = UserProfile(**profile_dict)

    # Add the liked or disliked movie to the correct list
    if liked_movie and liked_movie not in user_profile.liked_movies:
        user_profile.liked_movies.append(liked_movie)
        message = f"OK. I've added '{liked_movie}' to your liked movies."
    elif disliked_movie and disliked_movie not in user_profile.disliked_movies:
        user_profile.disliked_movies.append(disliked_movie)
        message = f"OK. I've noted that you disliked '{disliked_movie}'."
    else:
        message = "No changes made to your preferences."

    # Save the updated profile back to the session state
    tool_context.state["user_profile"] = user_profile.model_dump()
    logging.info(f"Updated user_profile state: {tool_context.state['user_profile']}")

    return {"status": "success", "message": message}
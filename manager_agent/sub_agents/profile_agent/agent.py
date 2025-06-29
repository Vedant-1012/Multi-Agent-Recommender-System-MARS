# manager_agent/sub_agents/profile_agent/agent.py

from google.adk.agents import Agent
from tools.movie_tools import get_movie_rating, update_user_preferences

profile_agent = Agent(
    name="profile_agent",
    description="Manages user profiles, including getting movie ratings and saving user preferences like likes or dislikes.",
    # --- FIX: Corrected instructions for the agent ---
    instruction="""
    You are an expert at managing a user's movie profile.

    - If the user asks for a movie's rating, use the `get_movie_rating` tool.
    - If the user expresses that they **liked** a movie, you MUST call the `update_user_preferences` tool, passing the movie's title to the `liked_movie` parameter.
    - If the user expresses that they **disliked** a movie, you MUST call the `update_user_preferences` tool, passing the movie's title to the `disliked_movie` parameter.

    Your response to the user should be the `message` from the tool's output.
    """,
    tools=[get_movie_rating, update_user_preferences],
    model="gemini-1.5-pro" # Using Pro for better reasoning on intent
)
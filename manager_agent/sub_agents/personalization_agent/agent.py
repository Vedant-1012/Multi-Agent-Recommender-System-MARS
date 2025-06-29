# /manager_agent/sub_agents/personalization_agent/agent.py

from google.adk.agents import Agent
from tools.personalization_tools import get_personalized_recommendations
import json

personalization_agent = Agent(
    name="personalization_agent",
    description="Generates deeply personalized movie recommendations for users with an established history, using a collaborative filtering model.",
    # --- FIX: Final, corrected instructions to parse the incoming request ---
    instruction="""
    You are a personalization expert. Your one job is to get movie recommendations for an established user.

    You will receive a JSON string in the user's `request`. This JSON contains the `user_id`.
    1.  You MUST parse this JSON to extract the `user_id`.
    2.  You MUST call the `get_personalized_recommendations` tool and pass the extracted `user_id` to it.
    3.  Once you have the list of titles from the tool, present it clearly to the user.

    For example: "Based on your viewing history, here are some movies I think you'll love:"
    """,
    tools=[get_personalized_recommendations],
    model="gemini-1.5-pro"  # Recommended: Use Pro for better instruction following and JSON parsing
)
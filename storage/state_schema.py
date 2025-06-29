# REASON: This updated version combines your desired fields (name, preferred_genres)
# with the Pydantic structure required by FastAPI and ADK for robust data
# handling and state management. The user_id is kept as it is essential
# for linking the profile to a specific user session in main.py.

from pydantic import BaseModel, Field
from typing import List, Optional

class UserProfile(BaseModel):
    """
    Data model for a user's profile, stored in the session state.
    Holds all information about a user's movie preferences.
    """
    user_id: str
    name: Optional[str] = None
    liked_movies: List[str] = Field(default_factory=list)
    disliked_movies: List[str] = Field(default_factory=list)
    preferred_genres: List[str] = Field(default_factory=list)


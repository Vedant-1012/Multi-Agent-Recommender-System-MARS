# REASON: This is the core fix. It now converts the UserProfile object into a
# plain dictionary using .model_dump() before saving it to the session state.
# This respects the ADK's requirement for simple data types and prevents the silent freeze.

import logging
from typing import Optional, Dict, List

from google.adk.sessions import Session
from google.adk.sessions.base_session_service import BaseSessionService
from .state_schema import UserProfile

# In-memory store for this example. In a real application, you'd use a database.
_STORE: Dict[str, Session] = {}

class PersistentSessionService(BaseSessionService):
    """Manages user sessions, ensuring each has a UserProfile."""

    async def get_session(self, app_name: str, user_id: str, session_id: str) -> Optional[Session]:
        key = f"{user_id}:{session_id}"
        if key in _STORE:
            logging.info(f"SESSION: Retrieved session for key {key}")
            return _STORE.get(key)
        return None

    async def create_session(self, app_name: str, user_id: str, session_id: str, state: Optional[dict] = None) -> Session:
        key = f"{user_id}:{session_id}"
        if key not in _STORE:
            logging.info(f"SESSION: Creating new session for key {key}")
            
            # 1. Create the UserProfile Pydantic object.
            profile_object = UserProfile(user_id=user_id)
            
            # 2. Convert the object to a dictionary before storing it.
            initial_state = {"user_profile": profile_object.model_dump()}
            
            _STORE[key] = Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=initial_state
            )
        return _STORE[key]

    async def update_session(self, session: Session) -> Session:
        key = f"{session.user_id}:{session.id}"
        logging.info(f"SESSION: Updating session for key {key}")
        _STORE[key] = session
        return session

    async def delete_session(self, app_name: str, user_id: str, session_id: str) -> None:
        key = f"{user_id}:{session_id}"
        if key in _STORE:
            logging.info(f"SESSION: Deleting session for key {key}")
            del _STORE[key]

    async def list_sessions(self, app_name: str, user_id: str) -> List[Session]:
        user_sessions = [s for s in _STORE.values() if s.user_id == user_id and s.app_name == app_name]
        logging.info(f"SESSION: Found {len(user_sessions)} sessions for user {user_id}")
        return user_sessions

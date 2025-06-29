# REASON: This file is reverted to a simple version that is compatible with
# your ADK environment. It no longer contains the `lifespan` or `app_context`
# logic, as the new lazy loading pattern makes them unnecessary.

import uuid
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from google.adk.runners import Runner
from google.genai import types
import os
from dotenv import load_dotenv

# --- Local Imports ---
load_dotenv()
from manager_agent.agent import root_agent
from storage.session_service import PersistentSessionService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - LOG - %(message)s')

# --- Application Setup ---
app = FastAPI(title="Movie Chatbot API")
session_service = PersistentSessionService()

# --- FIX: The runner is now simple again ---
runner = Runner(
    agent=root_agent,
    app_name="MovieChatbot",
    session_service=session_service
)

# --- API Data Models ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handles chat requests by running them through the agent graph."""
    
    await session_service.create_session("MovieChatbot", request.user_id, request.session_id)
        
    message = types.Content(role="user", parts=[types.Part(text=request.message)])
    final_response = ""

    try:
        print(f"\nINFO: Running agent for user '{request.user_id}' with message: '{request.message}'")
        for event in runner.run(user_id=request.user_id, session_id=request.session_id, new_message=message):
            if event.is_final_response() and event.content:
                final_response = event.content.parts[0].text
                break
    except Exception as e:
        import traceback
        print(f"ERROR: An error occurred during agent execution: {e}")
        traceback.print_exc()
        final_response = "I'm sorry, I ran into an internal error. Please try a different question."

    print(f"INFO: Final response for user '{request.user_id}': '{final_response}'")
    return ChatResponse(response=final_response, session_id=request.session_id)

@app.get("/")
def read_root():
    return {"message": "Movie Chatbot API is running. Send POST requests to /chat."}

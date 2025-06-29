import gradio as gr
import requests
import uuid
import time

# --- Configuration ---
# This is the URL where your FastAPI backend is running.
API_URL = "http://127.0.0.1:8000/chat"

# --- UI Customization ---
# A more professional and modern theme for our chatbot.
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Readex Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#F1F5F9",
    body_background_fill_dark="#0F172A",
    button_primary_background_fill_dark="*primary_500",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="*shadow_md",
    body_text_color_subdued_dark="#94A3B8"
)

# --- Gradio Application Logic ---

def get_session_ids(session_state):
    """Initializes and retrieves user and session IDs."""
    if not session_state.get("user_id"):
        session_state["user_id"] = str(uuid.uuid4())
        session_state["session_id"] = str(uuid.uuid4())
    return session_state.get("user_id"), session_state.get("session_id")

def chat_with_bot(message, history, session_state):
    """
    This function is called every time the user sends a message.
    It communicates with your FastAPI backend and yields the response.
    """
    user_id, session_id = get_session_ids(session_state)
    
    print(f"Calling API for user '{user_id}' in session '{session_id}'...")

    try:
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
        }
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        bot_response = response.json().get("response", "Sorry, I encountered an error.")
        
        # Simulate a "typing" effect for a better UX
        for i in range(len(bot_response)):
            time.sleep(0.01)
            yield bot_response[: i+1]
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        error_message = "I'm having trouble connecting to my brain right now. Please make sure the backend server is running and try again."
        for i in range(len(error_message)):
            time.sleep(0.01)
            yield error_message[: i+1]

# --- Build the Gradio Interface using Blocks for customization ---

with gr.Blocks(theme=theme, css="#chatbot { min-height: 600px; } footer { display: none !important; }") as demo:
    # State object to hold session information (user_id, session_id)
    session_state = gr.State({})

    with gr.Row():
        gr.Markdown(
            """
            # 🎬 Movie Recommender AI
            Your personal multi-agent assistant for discovering films.
            """
        )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                avatar_images=(None, "https://i.imgur.com/3g2Q41J.png"),
                label="MovieBot",
            )
            # Use a custom ChatInterface implementation
            with gr.Row():
                txt = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Ask me to recommend a movie, get a rating, or tell me what you like!",
                    container=False,
                )
                submit_btn = gr.Button("▶", variant="primary", scale=1, min_width=0)
        
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Try these examples:")
            examples = gr.Examples(
                examples=[
                    "Can you recommend a movie like 'The Matrix'?",
                    "I loved 'Parasite', can you save that to my profile?",
                    "What is the plot of 'Blade Runner 2049'?",
                    "What is the rating for 'Pulp Fiction'?"
                ],
                inputs=txt,
                label=""
            )

    def on_submit(message, history, session_state):
        """Handle submission: update chatbot history and stream response."""
        history.append([message, None])
        # We must return values for all outputs of the 'submit' event
        return [gr.update(value=""), history, session_state]

    def on_stream(history, session_state):
        """Handle streaming: get the last message and call the bot."""
        user_message = history[-1][0]
        history[-1][1] = ""
        for chunk in chat_with_bot(user_message, history, session_state):
            history[-1][1] = chunk
            yield [history, session_state]

    # Wire up the events
    txt.submit(on_submit, [txt, chatbot, session_state], [txt, chatbot, session_state], queue=False).then(
        on_stream, [chatbot, session_state], [chatbot, session_state]
    )
    submit_btn.click(on_submit, [txt, chatbot, session_state], [txt, chatbot, session_state], queue=False).then(
        on_stream, [chatbot, session_state], [chatbot, session_state]
    )


# --- Launch the Application ---
if __name__ == "__main__":
    demo.launch(debug=True) # Use debug=True for more detailed error messages

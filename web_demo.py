# web_demo.py

import os
import shutil
from typing import Tuple
import json
import threading

import gradio as gr

from src.chatbot import PDFChatBot
from scripts import pdf_extractor, chunker, build_index, section_rep_builder

# ---------------------------------------------------------------------
# Persistent user database (credentials + uploads + prompts)
# ---------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
USER_DB_PATH = os.path.join("data", "user_db.json")

# Re‑entrant lock to guard all reads/writes to the shared user DB in multi‑threaded Gradio
_DB_LOCK = threading.RLock()


def _load_user_db():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # default structure
    return {"users": {}}


def _save_user_db(db: dict):
    """Persist the in‑memory DB atomically."""
    with _DB_LOCK:
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)

DEFAULT_PROMPT = (
    "Answer the user's question based on the information provided in the document context below.\n"
    "Your response should reference the context clearly, but you may paraphrase or summarize appropriately."
)

# In‑memory view of the persistent database
_USER_DB = _load_user_db()
USERS = {u: info["password"] for u, info in _USER_DB["users"].items()}


def authenticate(username: str, password: str) -> bool:
    """Check if the provided credentials are valid."""
    return USERS.get(username) == password


def ensure_user_dir(username: str) -> str:
    """Create and return the directory for a specific user."""
    user_dir = os.path.join("data", "user_uploads", username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def login(username: str, password: str):
    if authenticate(username, password):
        return True, username, "Login successful."
    if username not in USERS:
        # Atomically create new user
        with _DB_LOCK:
            USERS[username] = password
            _USER_DB["users"][username] = {
                "password": password,
                "uploads": [],
                "prompts": [],
            }
            _save_user_db(_USER_DB)
        return True, username, "New user created and logged in."
    return False, "", "Invalid credentials."


def login_and_prepare(username: str, password: str):
    """
    Wrapper for the login flow that also controls component visibility
    and restores the last system prompt after a successful login.
    Returns:
        - logged‑in state (bool)
        - username (str)
        - login message (str)
        - update for the main interaction area (gr.update)
        - update for the prompt textbox (gr.update)
        - update for the existing‑PDF dropdown (gr.update)
    """
    success, uid, msg = login(username, password)

    # Toggle the main area
    main_area_update = gr.update(visible=success)

    # Restore the user's last prompt if available
    prompt_val = DEFAULT_PROMPT
    if success:
        prompts = _USER_DB["users"].get(uid, {}).get("prompts", [])
        if prompts:
            prompt_val = prompts[-1]
    prompt_update = gr.update(value=prompt_val)

    # Populate dropdown with user's previous uploads
    uploads = _USER_DB["users"].get(uid, {}).get("uploads", []) if success else []
    dropdown_update = gr.update(choices=[os.path.basename(u) for u in uploads],
                                value=(os.path.basename(uploads[0]) if uploads else None))

    return success, uid, msg, main_area_update, prompt_update, dropdown_update


def process_pdf(pdf_path: str) -> Tuple[list, list]:
    """Run the extraction/index pipeline for the given PDF."""
    extracted = pdf_extractor.extract_pdf_content(pdf_path)
    chunks = chunker.process_extracted_file(extracted)
    chunk_index = build_index.build_chunk_index(chunks)
    sections = section_rep_builder.build_section_reps(extracted["sections"], chunk_index)
    return sections, chunk_index


def load_pdf(pdf_file, system_prompt, username):
    if not username:
        return None, None, "Please log in first."
    if pdf_file is None:
        return None, None, "Please upload a PDF."
    user_dir = ensure_user_dir(username)
    dest_path = os.path.join(user_dir, os.path.basename(pdf_file.name))
    shutil.copy(pdf_file.name, dest_path)
    sections, chunk_index = process_pdf(dest_path)
    msg = f"Processed {os.path.basename(dest_path)}"
    # Record upload & system prompt for this user and persist
    with _DB_LOCK:
        user_record = _USER_DB["users"].setdefault(
            username,
            {"password": USERS[username], "uploads": [], "prompts": []},
        )
        if dest_path not in user_record["uploads"]:
            user_record["uploads"].append(dest_path)
        if system_prompt and system_prompt not in user_record["prompts"]:
            user_record["prompts"].append(system_prompt)
        _save_user_db(_USER_DB)
    return sections, chunk_index, msg


# Helper to load an existing PDF by name for the user
def load_existing_pdf(selected_name, system_prompt, username):
    if not username:
        return None, None, "Please log in first."
    if not selected_name:
        return None, None, "No previous PDF selected."
    user_dir = ensure_user_dir(username)
    pdf_path = os.path.join(user_dir, selected_name)
    if not os.path.exists(pdf_path):
        return None, None, "File not found."
    sections, chunk_index = process_pdf(pdf_path)
    msg = f"Processed {selected_name}"
    return sections, chunk_index, msg


def ask_question(question, sections, chunk_index, system_prompt, username, use_index):
    fine_only = not use_index 
    if not username:
        return "Please log in first."
    if sections is None or chunk_index is None:
        return "Please upload and process a PDF first."
    prompt = system_prompt or DEFAULT_PROMPT
    bot = PDFChatBot(sections, chunk_index, system_prompt=prompt)
    answer = bot.answer(question, fine_only=fine_only)
    answer = answer.replace('<|endoftext|><|im_start|>user',"=== System Prompt ===")
    answer = answer.replace('<|im_end|>\n<|im_start|>assistant','')
    answer = answer.replace('<|im_end|>','')
    return answer


with gr.Blocks() as demo:
    gr.Markdown("## QueryDoc Web Demo")

    # Login components
    with gr.Column():
        with gr.Row():
            login_user = gr.Textbox(label="Username")
            login_pass = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login", variant="primary")
        login_status = gr.Textbox(label="Login Status", interactive=False)

    # Main interaction area – hidden until login is successful
    with gr.Column(visible=False) as main_area:
        with gr.Row():
            with gr.Row():
                # Left column – previously uploaded files
                with gr.Column():
                    gr.Markdown("### Previously Uploaded PDFs")
                    existing_dropdown = gr.Dropdown(label="Select a PDF", choices=[])
                    load_existing_btn = gr.Button("Load Selected", variant="secondary")

                # Right column – new upload
                with gr.Column():
                    gr.Markdown("### Upload PDF")
                    gr.Markdown("- Upload a PDF file to query.")
                    pdf_input = gr.File(label="PDF File", file_types=[".pdf"])
                    load_btn = gr.Button("Load PDF", variant="primary")

            # PDF status textbox below the two columns
            status = gr.Textbox(label="PDF Status", interactive=False)
        gr.Markdown("### System Prompt")
        gr.Markdown("- Customize the system prompt for the PDF query.")
        gr.Markdown("- The system prompt will be used to guide the response.")
        gr.Markdown("- The system prompt will be saved for future use under same Username/Password.")
        prompt_input = gr.Textbox(label="System Prompt", lines=10, value=DEFAULT_PROMPT)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Ask a Question")
                gr.Markdown("- Ask a question based on the uploaded PDF.")
                gr.Markdown("- Check **Coarse-to-Fine Search** to enable Table of Contents based search.")
                question_input = gr.Textbox(label="Question")
                use_index = gr.Checkbox(label="Coarse-to-Fine Search", value=False)
                ask_btn = gr.Button("Ask", variant="primary")
        gr.Markdown("### Answer")
        answer_output = gr.Textbox(label="Answer")

    logged_in_state = gr.State(False)
    username_state = gr.State("")
    sections_state = gr.State()
    index_state = gr.State()

    login_btn.click(
        login_and_prepare,
        inputs=[login_user, login_pass],
        outputs=[logged_in_state, username_state, login_status, main_area, prompt_input, existing_dropdown],
    )
    load_btn.click(load_pdf, inputs=[pdf_input, prompt_input, username_state], outputs=[sections_state, index_state, status])
    load_existing_btn.click(
        load_existing_pdf,
        inputs=[existing_dropdown, prompt_input, username_state],
        outputs=[sections_state, index_state, status]
    )
    question_input.submit(ask_question, inputs=[question_input, sections_state, index_state, prompt_input, username_state], outputs=answer_output)
    ask_btn.click(ask_question, inputs=[question_input, sections_state, index_state, prompt_input, username_state, use_index], outputs=answer_output)

if __name__ == "__main__":
    demo.launch(server_port=30987, server_name="0.0.0.0")

# web_demo.py

import os
import shutil
from typing import Tuple

import gradio as gr

from src.chatbot import PDFChatBot
from scripts import pdf_extractor, chunker, build_index, section_rep_builder

DEFAULT_PROMPT = (
    "Answer the user's question based on the information provided in the document context below.\n"
    "Your response should reference the context clearly, but you may paraphrase or summarize appropriately."
)

# Simple hard-coded user database
USERS = {"admin": "password"}


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
    return False, "", "Invalid credentials."


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
    return sections, chunk_index, msg


def ask_question(question, sections, chunk_index, system_prompt, username):
    if not username:
        return "Please log in first."
    if sections is None or chunk_index is None:
        return "Please upload and process a PDF first."
    prompt = system_prompt or DEFAULT_PROMPT
    bot = PDFChatBot(sections, chunk_index, system_prompt=prompt)
    return bot.answer(question)


with gr.Blocks() as demo:
    gr.Markdown("## QueryDoc Web Demo")

    # Login components
    with gr.Box():
        with gr.Row():
            login_user = gr.Textbox(label="Username")
            login_pass = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
        login_status = gr.Textbox(label="Login Status", interactive=False)

    with gr.Row():
        pdf_input = gr.File(label="PDF File", file_types=[".pdf"])
        prompt_input = gr.Textbox(label="System Prompt", value=DEFAULT_PROMPT)
        load_btn = gr.Button("Load PDF")
    status = gr.Textbox(label="Status", interactive=False)
    question_input = gr.Textbox(label="Question")
    answer_output = gr.Textbox(label="Answer")

    logged_in_state = gr.State(False)
    username_state = gr.State("")
    sections_state = gr.State()
    index_state = gr.State()

    login_btn.click(login, inputs=[login_user, login_pass], outputs=[logged_in_state, username_state, login_status])
    load_btn.click(load_pdf, inputs=[pdf_input, prompt_input, username_state], outputs=[sections_state, index_state, status])
    question_input.submit(ask_question, inputs=[question_input, sections_state, index_state, prompt_input, username_state], outputs=answer_output)

if __name__ == "__main__":
    demo.launch()

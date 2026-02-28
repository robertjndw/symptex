import base64
import json
import logging
import re
import uuid
from pathlib import Path

import requests
import streamlit as st

# Constants
API_URL = "http://host.docker.internal:8000/api/v1"

#todo update with Ollama models
AVAILABLE_MODELS = [
    "gemma-3-27b-it",
    "llama-3.3-70b-instruct",
    "llama-3.1-sauerkrautlm-70b-instruct",
    "qwq-32b",
    "mistral-large-instruct",
    "qwen3-235b-a22b"
]
PATIENT_ROLES = ["default", "alzheimer", "schwerhoerig", "verdraengung"]
TALKATIVENESS_LEVELS = ["kurz angebunden", "ausgewogen", "ausschweifend"]

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def init_session_state() -> None:
    """Initialize Streamlit session state variables"""
    if "condition" not in st.session_state:
        st.session_state.model = "qwen3-235b-a22b"
        st.session_state.condition = "alzheimer"
        st.session_state.talkativeness = "kurz angebunden"
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_patient_image() -> str:
    """Load and convert patient image to base64"""
    def img_to_base64(image_path: Path) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_path = Path(__file__).parent / "assets" / "anna_zank.png"
    return img_to_base64(img_path)

def setup_header_layout() -> None:
    """Configure header layout"""
    st.set_page_config(page_title="Symptex", page_icon="🤖")
    st.markdown("""
        <style>
            .patient-image {
                width: 90px;
                height: 90px;
                border-radius: 50%;
                object-fit: cover;
            }
            .title-container {
                display: flex;
                align-items: center;
            }
            .header-section {
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

def create_header(img_base64: str) -> None:
    """Create the header with patient image and name"""
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 3.4])
    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" 
                     class="patient-image">
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.title("Anna Zank")
    st.markdown('</div>', unsafe_allow_html=True)

def display_patient_info() -> None:
    """Display patient information box"""
    st.info(
        """**Fallvignette**:

Notfallmäßige Vorstellung mit dem RTW bei Sturz auf die rechte Hüfte im häuslichen Umfeld vor ca. 3 Stunden. Deutliche Druckdolenz über der rechten Hüfte, pDMS intakt.

**Patientenstammdaten**:

* Alter: 89 Jahre
* Geburtsdatum: 01.09.1935
* Ethnie: kaukasisch
* BMI: 20,5"""
)

def setup_sidebar() -> None:
    """Setup sidebar controls"""
    st.sidebar.selectbox("Modell", options=AVAILABLE_MODELS, key="model")
    st.sidebar.selectbox("Patientenrolle", options=PATIENT_ROLES, key="condition")
    st.sidebar.selectbox("Gesprächsverhalten", options=TALKATIVENESS_LEVELS, key="talkativeness")

def handle_chat_reset() -> None:
    """Handle chat reset functionality"""
    try:
        # Send request to clear db for this session
        response = requests.post(f"{API_URL}/reset/{st.session_state.session_id}")
        if response.status_code == 200:
            # Generate new session ID
            st.session_state.session_id = str(uuid.uuid4())
            # Clear frontend messages
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Error resetting chat memory")
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        st.error(f"Could not reset chat memory: {str(e)}")

def handle_chat_eval() -> None:
    """Handle chat rating functionality."""
    if not st.session_state.messages:
        st.warning("Der Chat enthält noch keine Nachrichten zur Bewertung.")
        return

    try:
        messages = [
            {"role": msg["role"], "output": msg["output"]} for msg in st.session_state.messages
        ]

        # Create placeholder for evaluation response
        response_placeholder = st.chat_message("patient").markdown("")

        with st.spinner("Anamnese Feedback wird erstellt..."):
            with requests.post(f"{API_URL}/eval", json={"messages": messages}, stream=True) as response:
                if response.status_code == 200:
                    evaluation_text = process_llm_response(response, response_placeholder)
                    st.session_state.messages.append({
                        "role": "patient",
                        "output": evaluation_text,
                    })
                else:
                    st.error(f"Fehler bei der Bewertung (Status: {response.status_code})")

    except Exception as e:
        logger.error(f"Error evaluating chat: {str(e)}")
        st.error(f"Fehler bei der Bewertung: {str(e)}")

def process_llm_response(
    response: requests.Response,
    response_placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, list[dict]]:
    """Process streaming response from LLM, strip <think> tags, and capture attach_docs event."""
    streamed_text = ""
    buffer = ""
    think_tags_removed = False
    pdf_docs: list[dict] = []

    for chunk in response.iter_content(chunk_size=None):
        buffer = _accumulate_chunk(buffer, chunk)
        if not buffer:
            continue

        buffer, think_tags_removed, waiting_for_think_end = _strip_think_tags_if_needed(
            buffer, think_tags_removed
        )
        if waiting_for_think_end:
            # we don't show anything until </think> is seen
            continue

        is_event, pdf_docs, buffer = _handle_json_event_if_any(buffer, pdf_docs)
        if is_event:
            # JSON events (like attach_docs) are not shown as text
            continue

        streamed_text, buffer, think_tags_removed = _consume_text_buffer(
            buffer, streamed_text, think_tags_removed, response_placeholder
        )

    # Final render (without cursor, etc.)
    response_placeholder.markdown(streamed_text)
    return streamed_text, pdf_docs

def render_vorbefunde_docs(pdf_docs):
    if not pdf_docs:
        return

    # You can change this to whatever UI you like (expander, separate chat message, etc.)
    with st.expander("📎 Vorbefunde (PDFs anzeigen)"):
        for i, doc in enumerate(pdf_docs, start=1):
            filename = doc.get("filename", f"vorbefund_{i}.pdf")
            content_b64 = doc.get("content_b64", "")

            if not content_b64:
                continue

            pdf_bytes = base64.b64decode(content_b64)

            # Download button
            st.download_button(
                label=f"📄 {filename} herunterladen",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"download_{filename}_{i}",
            )

            # Optional inline viewer
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{content_b64}"
                    width="700" height="900"
                    type="application/pdf"></iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

def _accumulate_chunk(buffer: str, chunk: bytes) -> str:
    if not chunk:
        return buffer
    return buffer + chunk.decode(errors="ignore")


def _strip_think_tags_if_needed(
    buffer: str, think_tags_removed: bool
) -> tuple[str, bool, bool]:
    """
    Remove <think>...</think> block once, if present and complete.
    Returns (new_buffer, think_tags_removed, waiting_for_closing_tag).
    """
    if think_tags_removed or "<think>" not in buffer:
        return buffer, think_tags_removed, False

    # Wait for closing tag before stripping
    if "</think>\n" not in buffer:
        # still waiting for the rest of the think block
        return buffer, think_tags_removed, True

    # Remove the <think>...</think>\n (and an optional extra newline)
    buffer = re.sub(r'^<think>[\s\S]*?</think>\n\n?', '', buffer)
    return buffer, True, False


def _handle_json_event_if_any(
    buffer: str, pdf_docs: list[dict]
) -> tuple[bool, list[dict], str]:
    """
    Detect and handle JSON events in the buffer.
    Returns (is_event, updated_pdf_docs, new_buffer).
    """
    stripped = buffer.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False, pdf_docs, buffer

    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        # Not valid JSON → treat as normal text
        return False, pdf_docs, buffer

    # attach_docs event with PDFs from backend
    if event.get("event") == "attach_docs":
        pdf_docs = event.get("docs", [])
        return True, pdf_docs, ""  # clear buffer

    # Other events could be handled here in the future
    return True, pdf_docs, ""  # event handled, but not shown as text


def _consume_text_buffer(
    buffer: str,
    streamed_text: str,
    think_tags_removed: bool,
    response_placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, str, bool]:
    """
    Append current buffer to streamed_text (if appropriate) and update the UI.
    Returns (new_streamed_text, new_buffer, new_think_tags_removed).
    """
    if not buffer:
        return streamed_text, buffer, think_tags_removed

    # Only display when either:
    # - we've already removed think tags, or
    # - there is no <think> at all in the remaining buffer.
    if think_tags_removed or "<think>" not in buffer:
        streamed_text += buffer
        buffer = ""
        if not think_tags_removed:
            think_tags_removed = True
        response_placeholder.markdown(streamed_text)

    return streamed_text, buffer, think_tags_removed

def main() -> None:
    """Main application function"""
    setup_header_layout()
    img_base64 = load_patient_image()
    init_session_state()
    
    create_header(img_base64)
    display_patient_info()
    setup_sidebar()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "output" in message:
                st.markdown(message["output"])

    # Handle user input
    if prompt := st.chat_input("Fange hier ein Gespräch an..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "output": prompt})

        data = {
            "message": prompt,
            "model": st.session_state.model,
            "condition": st.session_state.condition,
            "talkativeness": st.session_state.talkativeness,
            "patient_file_id": 3, # Anna Zank
            "session_id": st.session_state.session_id
        }

        with st.spinner("Denkt nach..."):
            response_placeholder = st.chat_message("assistant").markdown("")
            with requests.post(API_URL + "/chat", json=data, stream=True) as response:

                if response.status_code == 200:
                    streamed_text, pdf_docs = process_llm_response(response, response_placeholder)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "output": streamed_text,
                        "docs": pdf_docs,
                    })

                    # Optionally render the PDFs right away:
                    if pdf_docs:
                        render_vorbefunde_docs(pdf_docs)  # function from earlier answer
                else:
                    st.error("An error occurred while processing your message.")

    # Add sidebar buttons
    if st.sidebar.button("Chat zurücksetzen", use_container_width=True):
        handle_chat_reset()
    if st.sidebar.button("Chat bewerten", use_container_width=True):
        handle_chat_eval()

if __name__ == "__main__":
    main()


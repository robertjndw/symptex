import base64
import binascii
import json
import logging
import uuid
from pathlib import Path

import requests
import streamlit as st

# Constants
API_URL = "http://host.docker.internal:8000/api/v1"
PATIENT_ROLES = ["default", "alzheimer", "schwerhoerig", "verdraengung"]
TALKATIVENESS_LEVELS = ["kurz angebunden", "ausgewogen", "ausschweifend"]

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=30)
def fetch_model_catalog() -> tuple[list[str], str | None, str | None, str | None]:
    """Fetch model metadata from backend and normalize it for the UI."""
    endpoint = f"{API_URL}/available-models"
    try:
        response = requests.get(endpoint, timeout=5)
    except requests.RequestException as exc:
        logger.error("Could not fetch model catalog from %s: %s", endpoint, exc)
        return [], None, None, "Model list unavailable. Please try again later."

    if response.status_code != 200:
        logger.error(
            "Model catalog request failed with status %s: %s",
            response.status_code,
            response.text,
        )
        return [], None, None, f"Model list unavailable (status {response.status_code})."

    try:
        payload = response.json()
    except ValueError as exc:
        logger.error("Invalid JSON from model catalog endpoint: %s", exc)
        return [], None, None, "Model list unavailable due to invalid server response."

    provider = str(payload.get("provider", "")).strip() or None

    models: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if model_id and model_id not in models:
            models.append(model_id)

    if not models:
        return [], None, provider, "No models configured on backend."

    default_model = str(payload.get("default_model", "")).strip() or models[0]
    if default_model not in models:
        default_model = models[0]

    return models, default_model, provider, None


def init_session_state(default_model: str | None, available_models: list[str]) -> None:
    """Initialize Streamlit session state variables"""
    if "model" not in st.session_state:
        st.session_state.model = default_model or (available_models[0] if available_models else "")

    if available_models and st.session_state.model not in available_models:
        st.session_state.model = default_model or available_models[0]

    if "condition" not in st.session_state:
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

def setup_sidebar(
    available_models: list[str],
    model_fetch_error: str | None,
    provider: str | None,
) -> None:
    """Setup sidebar controls"""
    if provider:
        st.sidebar.caption(f"LLM Provider: {provider}")

    if model_fetch_error:
        st.sidebar.warning(model_fetch_error)

    if available_models:
        st.sidebar.selectbox("Modell", options=available_models, key="model")
    else:
        st.sidebar.selectbox("Modell", options=["No model available"], disabled=True)

    st.sidebar.selectbox("Patientenrolle", options=PATIENT_ROLES, key="condition")
    st.sidebar.selectbox("Gespraechsverhalten", options=TALKATIVENESS_LEVELS, key="talkativeness")

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
        if not st.session_state.get("model"):
            st.warning("Kein Modell ausgewaehlt.")
            return

        messages = [
            {"role": msg["role"], "output": msg["output"]} for msg in st.session_state.messages
        ]

        # Create placeholder for evaluation response
        response_placeholder = st.chat_message("patient").markdown("")

        with st.spinner("Anamnese Feedback wird erstellt..."):
            with requests.post(
                f"{API_URL}/eval",
                json={"model": st.session_state.model, "messages": messages},
                stream=True,
            ) as response:
                if response.status_code == 200:
                    evaluation_text, _ = process_llm_response(response, response_placeholder)
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
    """Process streaming response from LLM and capture attach_docs events."""
    streamed_text = ""
    buffer = ""
    pdf_docs: list[dict] = []

    for chunk in response.iter_content(chunk_size=None):
        buffer = _accumulate_chunk(buffer, chunk)
        if not buffer:
            continue

        is_event, pdf_docs, buffer = _handle_json_event_if_any(buffer, pdf_docs)
        if is_event:
            # JSON events (like attach_docs) are not shown as text
            continue

        streamed_text, buffer = _consume_text_buffer(buffer, streamed_text, response_placeholder)

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

            try:
                pdf_bytes = base64.b64decode(content_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                logger.error("Invalid base64 content for PDF '%s': %s", filename, exc)
                st.warning(f"PDF konnte nicht geladen werden: {filename}")
                continue

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


def _handle_json_event_if_any(
    buffer: str, pdf_docs: list[dict]
) -> tuple[bool, list[dict], str]:
    """
    Detect and handle JSON events in the buffer.
    Returns (is_event, updated_pdf_docs, new_buffer).
    """
    # TODO(protocol): Current event detection assumes the whole buffer is one JSON object.
    # If transport coalesces data (e.g., text + "\n{...}" in one chunk), control JSON can
    # leak into rendered chat text. Migrate to an explicit framed protocol (e.g. EVENT:/SSE/
    # NDJSON) and an incremental line-based parser.
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
    response_placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, str]:
    """
    Append current buffer to streamed_text and update the UI.
    Returns (new_streamed_text, new_buffer).
    """
    if not buffer:
        return streamed_text, buffer

    streamed_text += buffer
    buffer = ""
    response_placeholder.markdown(streamed_text)

    return streamed_text, buffer

def main() -> None:
    """Main application function"""
    setup_header_layout()
    img_base64 = load_patient_image()
    available_models, default_model, provider, model_fetch_error = fetch_model_catalog()
    init_session_state(default_model, available_models)
    
    create_header(img_base64)
    display_patient_info()
    setup_sidebar(available_models, model_fetch_error, provider)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "output" in message:
                st.markdown(message["output"])

    # Handle user input
    if prompt := st.chat_input("Fange hier ein Gespraech an...", disabled=not available_models):
        if not st.session_state.model:
            st.error("Kein Modell verfuegbar. Bitte Backend-Konfiguration pruefen.")
            return

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
            response_placeholder = st.chat_message("patient").markdown("")
            with requests.post(API_URL + "/chat", json=data, stream=True) as response:

                if response.status_code == 200:
                    streamed_text, pdf_docs = process_llm_response(response, response_placeholder)

                    st.session_state.messages.append({
                        "role": "patient",
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







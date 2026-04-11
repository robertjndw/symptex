import base64
import binascii
import html
import json
import logging
import os
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

DEFAULT_DEV_FRONTEND_MODELS = ["gpt-oss:120b-cloud", "llama3.2"]


def _parse_csv_models(raw_models: str) -> list[str]:
    parsed = [item.strip() for item in raw_models.split(",") if item.strip()]
    # Preserve order but remove duplicates.
    return list(dict.fromkeys(parsed))


@st.cache_data(ttl=30)
def load_dev_model_config() -> tuple[list[str], str | None, str | None]:
    """Load dev frontend model configuration from local env."""
    raw_models = os.getenv("DEV_FRONTEND_MODELS", "")
    available_models = _parse_csv_models(raw_models)
    if not available_models:
        available_models = list(DEFAULT_DEV_FRONTEND_MODELS)

    default_model = os.getenv("DEV_FRONTEND_DEFAULT_MODEL", "").strip() or available_models[0]
    if default_model not in available_models:
        default_model = available_models[0]

    dev_frontend_key = os.getenv("DEV_FRONTEND_KEY", "").strip() or None
    return available_models, default_model, dev_frontend_key


def init_session_state(
    default_model: str | None,
    available_models: list[str],
    dev_frontend_key: str | None,
) -> None:
    """Initialize Streamlit session state variables"""
    if "model" not in st.session_state:
        st.session_state.model = default_model or (available_models[0] if available_models else "")

    if available_models and st.session_state.model not in available_models:
        st.session_state.model = default_model or available_models[0]

    if "dev_frontend_key" not in st.session_state:
        st.session_state.dev_frontend_key = dev_frontend_key

    if dev_frontend_key:
        st.session_state.dev_frontend_key = dev_frontend_key

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
            .eval-card {
                border: 1px solid #d5d9dd;
                border-radius: 10px;
                padding: 0.9rem 1rem;
                margin: 0.6rem 0;
                background: linear-gradient(180deg, #ffffff 0%, #f7f9fb 100%);
            }
            .eval-card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.45rem;
            }
            .eval-card-category {
                font-weight: 700;
                color: #102a43;
            }
            .eval-card-score {
                font-weight: 700;
                color: #0b7285;
                background: #e6fcf5;
                border: 1px solid #96f2d7;
                border-radius: 999px;
                padding: 0.15rem 0.6rem;
                white-space: nowrap;
            }
            .eval-card-message {
                color: #1f2933;
                line-height: 1.4;
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
    dev_frontend_key: str | None,
) -> None:
    """Setup sidebar controls"""
    if not dev_frontend_key:
        st.sidebar.warning("DEV_FRONTEND_KEY is not configured. Dev chat endpoints will reject requests.")

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


def _dev_endpoint_headers() -> dict[str, str]:
    headers = {}
    if st.session_state.get("dev_frontend_key"):
        headers["X-Dev-Frontend-Key"] = st.session_state.dev_frontend_key
    return headers


def _try_parse_eval_payload(output_text: str) -> dict | None:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or not payload:
        return None

    for category, item in payload.items():
        if not isinstance(category, str) or not isinstance(item, dict):
            return None
        if "score" not in item or "message" not in item:
            return None
    return payload


def _render_eval_cards(payload: dict) -> None:
    cards: list[str] = []
    for category, item in payload.items():
        score = item.get("score", "?")
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        message = item.get("message", "")
        safe_message = html.escape(str(message)).replace("\n", "<br>")

        cards.append(
            f"""
            <div class="eval-card">
                <div class="eval-card-header">
                    <div class="eval-card-category">{html.escape(str(category))}</div>
                    <div class="eval-card-score">Score: {html.escape(str(score))}/5</div>
                </div>
                <div class="eval-card-message">{safe_message}</div>
            </div>
            """
        )

    st.markdown("".join(cards), unsafe_allow_html=True)


def render_chat_output(output_text: str) -> None:
    payload = _try_parse_eval_payload(output_text)
    if payload is None:
        st.markdown(output_text)
        return
    _render_eval_cards(payload)


def handle_chat_eval() -> None:
    """Handle chat rating functionality."""
    if not st.session_state.messages:
        st.warning("Der Chat enthält noch keine Nachrichten zur Bewertung.")
        return

    try:
        if not st.session_state.get("dev_frontend_key"):
            st.warning("DEV_FRONTEND_KEY fehlt. Bewertung ist nicht verfuegbar.")
            return

        if not st.session_state.get("model"):
            st.warning("Kein Modell ausgewaehlt.")
            return

        messages = [
            {"role": msg["role"], "output": msg["output"]} for msg in st.session_state.messages
        ]

        response_container = st.chat_message("patient")

        with st.spinner("Anamnese Feedback wird erstellt..."):
            response = requests.post(
                f"{API_URL}/dev/eval",
                json={"model": st.session_state.model, "messages": messages},
                headers=_dev_endpoint_headers(),
            )
            if response.status_code == 200:
                evaluation_text = response.text
                with response_container:
                    render_chat_output(evaluation_text)
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
    available_models, default_model, dev_frontend_key = load_dev_model_config()
    init_session_state(default_model, available_models, dev_frontend_key)
    
    create_header(img_base64)
    display_patient_info()
    setup_sidebar(available_models, dev_frontend_key)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "output" in message:
                render_chat_output(message["output"])

    # Handle user input
    chat_disabled = (not available_models) or (not st.session_state.get("dev_frontend_key"))
    if prompt := st.chat_input("Fange hier ein Gespraech an...", disabled=chat_disabled):
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
            "case_id": 3, # Anna Zank
            "session_id": st.session_state.session_id
        }

        with st.spinner("Denkt nach..."):
            response_placeholder = st.chat_message("patient").markdown("")
            with requests.post(
                API_URL + "/dev/chat",
                json=data,
                headers=_dev_endpoint_headers(),
                stream=True,
            ) as response:

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







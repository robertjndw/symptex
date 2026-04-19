# Symptex

A chatbot designed for medical students, simulating doctor-patient interactions with the goal of improving the user's medical
history-taking skills.

## Prerequisites

- [Docker](https://docs.docker.com/get-started/get-docker/)
- Running ILuVI PostgreSQL database (see ILuVI repository)
- Browser of your choice to interact with Symptex
- Access to an API key for the [KISSKI ChatAI service](https://kisski.gwdg.de/leistungen/2-02-llm-service/)

## Getting Started

1. In the project root (`symptex/.env`), create an `.env` file.
   This file is loaded into the `symptex` API container through `docker-compose.yml` (`env_file`).
   Add the following variables:
```env
# Required for local volume mount used by docker-compose
HOST_ANAMNESIS_PATH={path to Befunde}

# Optional for API database connection (defaults shown)
DATABASE_URL=postgresql://ilvi:ilvi@postgres:5432/ilvi
SYMPTEX_DATABASE_URL=postgresql://symptex:symptex@symptex-db:5432/symptex

# Required LLM provider selection
LLM_PROVIDER=chatai # or ollama

# Optional: enable development-only endpoints (/api/v1/dev/chat, /api/v1/dev/eval)
# Defaults to disabled when omitted.
SYMPTEX_DEV_MODE=false

# Optional runtime fallback defaults when case SymptexConfig is missing/disabled/invalid.
# Allowed values:
# - SYMPTEX_DEFAULT_CONDITION: default, alzheimer, schwerhoerig, verdraengung
# - SYMPTEX_DEFAULT_TALKATIVENESS: kurz angebunden, ausgewogen, ausschweifend
SYMPTEX_DEFAULT_CONDITION=default
SYMPTEX_DEFAULT_TALKATIVENESS=ausgewogen

# Required for provider "chatai"
LLM_CHATAI_BASE_URL=https://chat-ai.academiccloud.de/v1
LLM_CHATAI_API_KEY={api_key}
LLM_CHATAI_MODELS=qwen3-235b-a22b,llama-3.3-70b-instruct
LLM_CHATAI_MODEL=qwen3-235b-a22b # required runtime model for /chat and /eval

# Required for provider "ollama"
LLM_OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_OLLAMA_MODELS=gpt-oss:120b-cloud,llama3.2
LLM_OLLAMA_MODEL=gpt-oss:120b-cloud # required runtime model for /chat and /eval

# Required only when SYMPTEX_DEV_MODE=true
DEV_FRONTEND_KEY={shared_secret_between_frontend_and_api}

# Optional for development frontend model picker (frontend service)
DEV_FRONTEND_MODELS=gpt-oss:120b-cloud,llama3.2
DEV_FRONTEND_DEFAULT_MODEL=gpt-oss:120b-cloud

# Optional LLM tuning (defaults shown)
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.8
LLM_MAX_RETRIES=2

# Required for ILuVI AnamDocs REST integration
ILUVI_API_BASE_URL={base_url_of_ilvi_backend}

# Optional (defaults shown)
FILE_SERVER_ROUTE=/static
ANAMDOCS_HTTP_TIMEOUT_SEC=10
ANAMDOCS_MAX_DOCS=10
ANAMDOCS_MAX_FILE_MB=10
ANAMDOCS_MAX_TOTAL_MB=40

# Optional: local-only debug login fallback for ILuVI session auth
ILUVI_DEBUG_LOGIN_ENABLED=false
ILUVI_DEBUG_LOGIN_TUM_ID=ADMIN1234
ILUVI_DEBUG_LOGIN_ROLE=admin
ILUVI_DEBUG_LOGIN_FIRST_NAME=Symptex
ILUVI_DEBUG_LOGIN_LAST_NAME=Debug

# Optional for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=insert_langsmith_key
```
If `DATABASE_URL` is omitted or set to an empty value, the API falls back to `postgresql://ilvi:ilvi@postgres:5432/ilvi`.
If `SYMPTEX_DATABASE_URL` is omitted or set to an empty value, the API falls back to `DATABASE_URL`.
`ILUVI_DEBUG_LOGIN_ENABLED=true` is intended for local development only. It requires ILuVI to run with
`ILVI_DEBUG=true` so `/auth/debug-login` is available. Do not enable this in production.

### Required environment variables

The `.env` example above contains both required and optional variables. This matrix shows what is strictly required.

| Scope | Required variables | Why |
|---|---|---|
| Always (LLM config) | `LLM_PROVIDER` | Selects active provider (`chatai` or `ollama`) used by `/api/v1/chat`, `/api/v1/eval`, and `/api/v1/chat/options`. |
| If `LLM_PROVIDER=chatai` | `LLM_CHATAI_BASE_URL`, `LLM_CHATAI_API_KEY`, `LLM_CHATAI_MODELS`, `LLM_CHATAI_MODEL` | Required to initialize ChatAI client and runtime model. |
| If `LLM_PROVIDER=ollama` | `LLM_OLLAMA_BASE_URL`, `LLM_OLLAMA_MODELS`, `LLM_OLLAMA_MODEL` | Required to initialize Ollama client and runtime model. |
| AnamDocs integration | `ILUVI_API_BASE_URL` | Required to fetch and download AnamDocs from ILuVI backend. |
| Dev endpoints enabled | `DEV_FRONTEND_KEY` (with `SYMPTEX_DEV_MODE=true`) | Required by `/api/v1/dev/chat` and `/api/v1/dev/eval` via `X-Dev-Frontend-Key`. |
| Docker local setup | `HOST_ANAMNESIS_PATH` | Required by `docker-compose` local volume mount for anamnesis files. |

Notes:
- `LLM_*_MODEL` must be one of the values listed in the corresponding `LLM_*_MODELS`.
- `DATABASE_URL` and `SYMPTEX_DATABASE_URL` are not strictly required in local default setup because fallback values exist.
- `SYMPTEX_DEFAULT_CONDITION` and `SYMPTEX_DEFAULT_TALKATIVENESS` are optional fallbacks used when case config is missing/invalid.
- `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_RETRIES`, `ANAMDOCS_*`, `LOG_LEVEL`, `SYMPTEX_CORS_ALLOW_ORIGINS`, and `SYMPTEX_EVAL_LOG_RAW` are optional tuning/ops settings.

2. Run `docker compose up --build` in the project's root directory.
3. Interact with Symptex locally through [Streamlit frontend URL](http://localhost:8501).

## API Endpoints

- Streamlit frontend base URL: <http://localhost:8501>
- API base URL: <http://localhost:8000>

### Endpoint reference

| Method | Path | Description | Request model / params | Notes |
|---|---|---|---|---|
| `GET` | `/` | Health-style root endpoint | none | Returns `{"message":"Hello, World!"}` |
| `POST` | `/api/v1/chat` | Runtime chat using case-specific Symptex config | `ChatRequest` | Streams plain-text model output |
| `GET` | `/api/v1/chat/options` | Returns allowed runtime options | none | Returns `models`, `conditions`, `talkativeness` |
| `GET` | `/api/v1/chat/history` | Returns stored chat history for one session/case | query: `session_id`, `case_id` | Response model: `ChatHistoryResponse` |
| `POST` | `/api/v1/eval` | Evaluates anamnesis chat quality | `RateRequest` | Preferred payload: `session_id` + `case_id`; legacy `messages` still supported |
| `POST` | `/api/v1/reset/{session_id}` | Deletes chat messages and session | path: `session_id` | Returns plain text status |
| `GET` | `/api/v1/config` | Reads Symptex config for a case | query: `caseId` | Returns 404 if no config exists |
| `POST` | `/api/v1/config` | Creates/updates Symptex config for a case | `SymptexConfigRequest` | Upsert semantics (`updated: true/false`) |
| `DELETE` | `/api/v1/config/{case_id}` | Deletes Symptex config for a case | path: `case_id` | Idempotent success message if config is missing |
| `POST` | `/api/v1/dev/chat` | Development-only chat with explicit model/condition/talkativeness | `DevChatRequest` | Available only with `SYMPTEX_DEV_MODE=true`, requires header `X-Dev-Frontend-Key` |
| `POST` | `/api/v1/dev/eval` | Development-only eval with explicit model | `DevRateRequest` | Available only with `SYMPTEX_DEV_MODE=true`, requires header `X-Dev-Frontend-Key` |

### Runtime values returned by `/api/v1/chat/options`

- `conditions`: `default`, `alzheimer`, `schwerhoerig`, `verdraengung`
- `talkativeness`: `kurz angebunden`, `ausgewogen`, `ausschweifend`
- `models`: derived from configured LLM provider model list

## Models

### API request/response models (Pydantic)

| Model | Fields |
|---|---|
| `ChatRequest` | `message: str`, `case_id: int`, `session_id: str` |
| `RateRequest` | `session_id: str | None`, `case_id: int | None`, `messages: list | None` |
| `ChatHistoryMessageResponse` | `id: int`, `role: str`, `content: str`, `timestamp: datetime` |
| `ChatHistoryResponse` | `session_id: str`, `case_id: int`, `messages: list[ChatHistoryMessageResponse]` |
| `SymptexConfigRequest` | `caseId: int`, `model: str`, `talkativeness: str`, `condition: str` (strict types, unknown fields forbidden) |
| `DevChatRequest` | `message: str`, `model: str`, `condition: str`, `talkativeness: str`, `case_id: int`, `session_id: str` |
| `DevRateRequest` | `model: str`, `messages: list` |

### Database models (SQLAlchemy)

Symptex reads ILuVI domain data from the ILuVI DB and stores chat/config state in the Symptex DB.

#### ILuVI DB models (`api/app/db/models.py`)

| Model | Main fields |
|---|---|
| `PatientFile` | `id`, `first_name`, `last_name`, `birth_date`, `height`, `weight`, `gender_identity`, `gender_medical`, `ethnic_origin`; relations: `anamneses`, `cases` |
| `Anamnesis` | `id`, `category`, `answer`, `patient_file_id`; relations: `patient_file`, `anam_docs` |
| `AnamDoc` | `id`, `category`, `original_name`, `storage_key` (unique), `anamnesis_id` (nullable FK with `ondelete=SET NULL`) |
| `Case` | `id`, `created_at`, `updated_at`, `deleted_at`, `title`, `treatment_reason`, `start_date`, `due_date`, `marked`, `time_budget`, `money_budget`, `diagnosis`, `treatment`, `is_draft`, `lecture_id`, `patient_file_id`; relation: `patient_file` |

#### Symptex DB models (`api/app/db/symptex_models.py`)

| Model | Main fields |
|---|---|
| `ChatSession` | `id` (string PK), `patient_file_id`, `case_id`, `created_at`; relation: `messages` |
| `ChatMessage` | `id`, `session_id` (FK -> `chat_sessions.id`), `role`, `content`, `timestamp`; relation: `session` |
| `SymptexConfig` | `id`, `created_at`, `updated_at`, `deleted_at`, `case_id` (unique), `model`, `condition`, `talkativeness` |

## Features

- Simulation of multiple patient conditions in the context of medical history-taking: default, alzheimer, schwerhörig (hearing impairment), verdrängung (denial of symptoms)
- Configurable patient talkativeness levels/verbosity: kurz angebunden, ausgewogen, ausschweifend
- Provision of performance feedback for increased pedagogical value
- Multiple LLM models supported (see [KISSKI ChatAI models](https://docs.hpc.gwdg.de/services/saia/index.html))
- Chat session management through ILuVI PostgreSQL database
- ILuVI Patient file integration

## Project Structure

```
symptex/
│
├── api/
│   ├── app/                      # API logic
│   │   ├── main.py               # FastAPI entry point
│   │   ├── db/                   # Database models and connection
│   │   │   ├── db.py             # Database configuration
│   │   │   └── models.py         # SQLAlchemy models
│   │   └── routers/
│   │       └── chat.py           # Chat-specific routes
│   │
│   ├── chains/                   # Chain logic
│   │   ├── chat_chain.py         # Main chat chain definition
│   │   ├── eval_chain.py         # Evaluation chain for feedback
│   │   ├── prompts.py            # Behavior prompts for different conditions
│   │   ├── patient_data.py       # Patient data definitions for testing
│   │   └── formatting.py         # Patient data formatting utilities
│   │
│   ├── tests/                    # Test files
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── frontend.py               # Streamlit frontend
│   ├── requirements.txt          # Dependencies for Streamlit frontend
│   ├── assets/                   # Frontend assets (images, etc.)
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

TODOs: Connect the upload of Befunde with ILVI's backend

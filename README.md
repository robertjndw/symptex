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

# Required LLM provider selection
LLM_PROVIDER=chatai # or ollama

# Required for provider "chatai"
LLM_CHATAI_BASE_URL=https://chat-ai.academiccloud.de/v1
LLM_CHATAI_API_KEY={api_key}
LLM_CHATAI_MODELS=qwen3-235b-a22b,llama-3.3-70b-instruct

# Required for provider "ollama"
LLM_OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_OLLAMA_MODELS=gpt-oss:120b-cloud,llama3.2

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
`ILUVI_DEBUG_LOGIN_ENABLED=true` is intended for local development only. It requires ILuVI to run with
`ILVI_DEBUG=true` so `/auth/debug-login` is available. Do not enable this in production.

2. Run `docker compose up --build` in the project's root directory.
3. Interact with Symptex locally through [Streamlit frontend URL](http://localhost:8501).

## Endpoints

- Streamlit frontend: <http://localhost:8501>
- API: <http://localhost:8000>

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

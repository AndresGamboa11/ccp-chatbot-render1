# app/settings.py
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

# Ruta ABSOLUTA al .env en la RAÍZ del proyecto (carpeta que contiene /app)
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"  # <-- siempre este .env, sin depender del cwd

class Settings(BaseSettings):
     # --- WhatsApp Cloud API ---
    WA_ACCESS_TOKEN: str | None = None
    WA_PHONE_NUMBER_ID: str | None = None
    WA_VERIFY_TOKEN: str | None = None
    WA_API_VERSION: str = "v21.0"  # ✅ versión por defecto para Graph API

    # --- Groq / Gemma ---
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "gemma2-9b-it"

    # --- Hugging Face (embeddings) ---
    HF_API_TOKEN: str | None = None
    HF_EMBED_MODEL: str = "intfloat/multilingual-e5-small"

    # --- Qdrant / Chroma ---
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "ccp_docs"

    # --- Otros opcionales ---
    ENV: str = "local"

@lru_cache()
def get_settings() -> Settings:
    # Carga .env explícito (útil en scripts sueltos)
    load_dotenv(dotenv_path=ENV_PATH)
    return Settings()

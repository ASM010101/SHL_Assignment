"""
Configuration management for SHL Assessment Recommender.

Loads settings from environment variables with sensible defaults.
Uses pydantic-settings pattern for type-safe configuration.

Design Decision: Centralized config avoids scattered env lookups.
Improves: Determinism (consistent settings), Interview defensibility.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (development mode)
load_dotenv()

# ─── Base Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

# ─── LLM Configuration ──────────────────────────────────────────────────────

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google_gemini")  # "google_gemini" or "ollama"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Ollama-specific settings
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "glm-5.2:cloud")

# ─── Embedding Configuration ────────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Retrieval Configuration ────────────────────────────────────────────────

FAISS_TOP_K: int = int(os.getenv("FAISS_TOP_K", "30"))
KEYWORD_TOP_K: int = int(os.getenv("KEYWORD_TOP_K", "20"))
FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "10"))

# ─── Conversation Configuration ─────────────────────────────────────────────

MAX_TURNS: int = int(os.getenv("MAX_TURNS", "8"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
MAX_CLARIFICATION_ROUNDS: int = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))

# ─── Server Configuration ───────────────────────────────────────────────────

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ─── Data Paths ──────────────────────────────────────────────────────────────

CATALOG_PATH: str = os.getenv("CATALOG_PATH", str(DATA_DIR / "shl_product_catalog.json"))
ENRICHED_CATALOG_PATH: str = os.getenv("ENRICHED_CATALOG_PATH", str(DATA_DIR / "enriched_catalog.json"))
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index.bin"))

# ─── Test Type Mapping ──────────────────────────────────────────────────────
# Maps catalog 'keys' field to short test type codes used in API responses.
# Derived from sample conversations: K=Knowledge, P=Personality, A=Ability,
# C=Competencies, S=Simulations, B=Biodata/SJT, D=Development

TEST_TYPE_MAP: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

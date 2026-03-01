import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
ENDEE_BASE_URL = "http://localhost:8080"

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "..", "cache")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K = 6
CONTEXT_CHUNKS = 4
CHAT_MEMORY = 3
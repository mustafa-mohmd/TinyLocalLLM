import os

APP_NAME = "LocalLLM Chat"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHAT_DIR = os.path.join(BASE_DIR, "chat_history")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLES_DIR = os.path.join(BASE_DIR, "styles")

DEFAULT_MODEL_NAME = "gemma3:4b"

SETTINGS_FILE = os.path.join(CHAT_DIR, "settings.json")
#fine-tuning
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_REPEAT_PENALTY = 1.1

DEFAULT_BATCH = 512
DEFAULT_THREADS = max(1, (os.cpu_count() or 6) - 2)

DEFAULT_OLLAMA_HOST = ""

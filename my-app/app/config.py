from pathlib import Path

class Settings:
    # API Config
    API_TITLE = "LangChain Server"
    API_VERSION = "1.0"
    
    # Model Config
    OLLAMA_URL = "https://b198-112-109-90-2.ngrok-free.app"
    LLM_MODEL = "qwen2"
    EMBED_MODEL = "nomic-embed-text"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    PDF_DIR = BASE_DIR / "pdf"
    DB_DIR = BASE_DIR / "db"
    
    # Processing Config
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 20
    
    def setup(self):
        self.PDF_DIR.mkdir(exist_ok=True)
        self.DB_DIR.mkdir(exist_ok=True)

settings = Settings()
settings.setup()
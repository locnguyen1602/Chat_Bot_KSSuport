from pathlib import Path
from urllib.parse import quote_plus


class Settings:
    # API Config
    API_TITLE = "LangChain Server"
    API_VERSION = "1.0"

    # Model Config
    OLLAMA_URL = "https://35f7-112-109-90-2.ngrok-free.app/"
    LLM_MODEL = "qwen2"
    EMBED_MODEL = "nomic-embed-text"

    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    PDF_DIR = BASE_DIR / "pdf"
    DB_DIR = BASE_DIR / "db"
    CHROMA_DIR = DB_DIR / "chroma"  # Directory for ChromaDB
    STORAGE_DIR = "storage"
    DATA_DIR = BASE_DIR / "data"
    CHAT_HISTORY_DIR = DATA_DIR / "chat_history"

    # Processing Config
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 20

    # Collection Config
    COLLECTION_NAME = "pdf_collection"

    # Database settings
    DB_HOST = "localhost"
    DB_PORT = "3306"
    DB_USERNAME = "root"
    DB_PASSWORD = "123qwe"
    DB_NAME = "{0}"  # Placeholder for database name

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    CHAT_HISTORY_DIR.mkdir(exist_ok=True)

    # Chat history settings
    CHAT_HISTORY_FILE = CHAT_HISTORY_DIR / "history.json"

    # WeatherAPI Config
    WEATHER_API_KEY = "920d37982e004d89a1965418242209"

    @property
    def DB_CONNECTION_STRING(self):
        """Returns the database connection string with encoded password"""
        encoded_password = quote_plus(self.DB_PASSWORD)
        return f"mysql+mysqlconnector://{self.DB_USERNAME}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def setup(self):
        self.PDF_DIR.mkdir(exist_ok=True, parents=True)
        self.DB_DIR.mkdir(exist_ok=True, parents=True)
        self.CHROMA_DIR.mkdir(exist_ok=True, parents=True)

settings = Settings()
settings.setup()

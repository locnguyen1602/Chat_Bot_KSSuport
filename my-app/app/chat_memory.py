from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import json
import os
from .config import settings


class ChatHistory:
    def __init__(self, history_file: Path = settings.CHAT_HISTORY_FILE):
        self.messages = []
        self.history_file = history_file
        # Create directory if not exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load chat history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            self.messages = []

    def _save_history(self):
        """Save chat history to file"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_message(self, role: str, content: str):
        """Add new message to history"""
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
            self.messages.append(message)
            self._save_history()
        except Exception as e:
            print(f"Error adding message: {e}")

    def get_messages(self) -> List[Dict]:
        """Get all messages"""
        return self.messages

    def get_recent(self, limit: int = 5) -> List[Dict]:
        """Get recent messages"""
        return self.messages[-limit:] if self.messages else []

    def clear(self):
        """Clear all messages"""
        try:
            self.messages = []
            self._save_history()
        except Exception as e:
            print(f"Error clearing history: {e}")


# Create single instance
chat_history = ChatHistory()

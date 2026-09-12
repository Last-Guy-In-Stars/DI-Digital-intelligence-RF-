import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, path=None):
        self.path = Path(path) if path else ROOT / "config.json"
        with open(self.path, encoding="utf-8") as f:
            self.data = json.load(f)

    def __getitem__(self, key):
        return self.data[key]

    @property
    def sqlite_path(self):
        return ROOT / self.data["sqlite_path"]

    @property
    def llm_path(self):
        return ROOT / self.data["llm_path"]

    @property
    def translator_path(self):
        return ROOT / self.data.get(
            "translator_path", "brain/models/qwen3-4b-q4_km.gguf")

    @property
    def embedder_path(self):
        return ROOT / self.data["embedder_path"]


def load(path=None):
    return Config(path)

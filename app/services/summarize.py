import threading
from typing import Protocol


class SummarizerLike(Protocol):
    def summarize(self, title: str, abstract: str) -> str: ...


class Summarizer:
    """BART summarizer; the model loads on first use."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_input_chars: int = 1024):
        self.model_name = model_name
        self.max_input_chars = max_input_chars
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline

                self._pipe = pipeline("summarization", model=self.model_name)
        return self._pipe

    def summarize(self, title: str, abstract: str) -> str:
        text = f"{title}. {abstract}"[: self.max_input_chars]
        result = self._load()(text, max_length=130, min_length=30, do_sample=False)
        return result[0]["summary_text"]

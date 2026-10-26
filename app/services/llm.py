import logging
import threading
from typing import Protocol

from app.config import Settings
from app.services.http import Http, HttpError

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"


class LLMError(Exception):
    """The LLM call failed or returned something unusable."""


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message]) -> str: ...


class _Usage:
    """Running token totals, so evals can report cost without parsing logs."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def add(self, model: str, prompt: int | None, completion: int | None) -> None:
        with self._lock:
            self.prompt_tokens += prompt or 0
            self.completion_tokens += completion or 0
        log.info(
            "llm usage model=%s prompt_tokens=%s completion_tokens=%s", model, prompt, completion
        )


class OpenAICompatClient:
    """Any server speaking POST {base_url}/chat/completions: OpenAI, Ollama, vLLM, ..."""

    def __init__(
        self,
        model: str,
        base_url: str = OPENAI_DEFAULT_BASE,
        api_key: str | None = None,
        http: Http | None = None,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.http = http or Http(timeout=60.0, retries=3)
        self.usage = _Usage()

    def complete(self, messages: list[Message]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        body = {"model": self.model, "messages": messages, "temperature": 0}
        try:
            data = self.http.post_json(self.url, body, headers)
            text = data["choices"][0]["message"]["content"]
        except HttpError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected LLM response shape: {e!r}") from e
        usage = data.get("usage") or {}
        self.usage.add(self.model, usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return text or ""


class AnthropicClient:
    def __init__(self, model: str, api_key: str, http: Http | None = None):
        self.model = model
        self.api_key = api_key
        self.http = http or Http(timeout=60.0, retries=3)
        self.usage = _Usage()

    def complete(self, messages: list[Message]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            body["system"] = system
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        try:
            data = self.http.post_json(ANTHROPIC_URL, body, headers)
            text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
        except HttpError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected LLM response shape: {e!r}") from e
        usage = data.get("usage") or {}
        self.usage.add(self.model, usage.get("input_tokens"), usage.get("output_tokens"))
        return text


def make_llm_client(settings: Settings) -> LLMClient | None:
    """Build the configured client, or None when the LLM backend is not configured.

    There is deliberately no default model name: LLM_MODEL must be set."""
    if not settings.llm_model:
        return None
    http = Http(timeout=settings.llm_timeout_seconds, retries=3)
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            return None
        return AnthropicClient(settings.llm_model, settings.anthropic_api_key, http)
    if not (settings.llm_api_key or settings.llm_base_url):
        return None  # a keyless server (Ollama) is fine, but only if its URL is given
    return OpenAICompatClient(
        settings.llm_model, settings.llm_base_url or OPENAI_DEFAULT_BASE, settings.llm_api_key, http
    )

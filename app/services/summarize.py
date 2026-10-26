import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol

from pydantic import BaseModel, ValidationError, field_validator

from app.schemas import CLAIM_FIELDS, Claim, PaperIn, PaperSummary
from app.services.llm import LLMClient, LLMError, Message

log = logging.getLogger(__name__)

BACKENDS = ("hf", "llm")
PROMPT_VERSION = "summ-v1"  # bump when the prompt or schema changes; part of every cache key
MAX_RETRIES = 2  # extra attempts after an invalid reply, each fed the validation error
MIN_EVIDENCE_CHARS = 12  # shorter quotes ("the") would be "verified" by almost any abstract
NOT_STATED = "not stated"


class SummarizeError(Exception):
    """One paper could not be summarized; the review carries on without it."""


class Summarizer(Protocol):
    id: str  # goes into the review cache key, so a different model never serves stale output

    def summarize(self, paper: PaperIn) -> PaperSummary: ...


class HFSummarizer:
    """BART summarizer; the model loads on first use."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_input_chars: int = 1024):
        self.model_name = model_name
        self.id = f"hf:{model_name}"
        self.max_input_chars = max_input_chars
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline

                self._pipe = pipeline("summarization", model=self.model_name)
        return self._pipe

    def summarize(self, paper: PaperIn) -> PaperSummary:
        text = f"{paper.title}. {paper.abstract}"[: self.max_input_chars]
        result = self._load()(text, max_length=130, min_length=30, do_sample=False)
        return PaperSummary(overview=result[0]["summary_text"])


SYSTEM_PROMPT = f"""You extract structured facts from the abstract of a research paper.
Reply with ONLY a JSON object with exactly these keys: "problem", "method", "results", \
"limitations". Each value is either
  - a non-empty list of claims, each {{"text": "<one-sentence claim>", "evidence": "<exact \
quote copied verbatim from the abstract that supports the claim>"}}, or
  - the string "{NOT_STATED}" if the abstract says nothing about that key.
Rules: use only the abstract; never guess or use outside knowledge. Abstracts often do not \
state limitations, so "{NOT_STATED}" is expected and correct there. Evidence must be copied \
character for character, contiguous, and short (a phrase or a sentence). No markdown, no \
commentary."""


class RawSummary(BaseModel):
    """The JSON the model must return, before the evidence is checked against the abstract."""

    problem: list[Claim] | Literal["not stated"]
    method: list[Claim] | Literal["not stated"]
    results: list[Claim] | Literal["not stated"]
    limitations: list[Claim] | Literal["not stated"]

    @field_validator(*CLAIM_FIELDS, mode="before")
    @classmethod
    def _normalize_not_stated(cls, v):
        if isinstance(v, str) and v.strip().rstrip(".").lower() == NOT_STATED:
            return NOT_STATED
        return v

    @field_validator(*CLAIM_FIELDS, mode="after")
    @classmethod
    def _claims_must_be_complete(cls, v):
        if v == NOT_STATED:
            return v
        if not v:
            raise ValueError(f'use "{NOT_STATED}" instead of an empty list')
        if any(not c.text.strip() or not c.evidence.strip() for c in v):
            raise ValueError("every claim needs non-empty text and evidence")
        return v


def parse_summary_json(reply: str) -> RawSummary:
    """Parse a model reply into a RawSummary; ValueError/ValidationError if it is not one."""
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")  # tolerate ```json fences and chatter
    if start < 0 or end < start:
        raise ValueError("no JSON object found in the reply")
    return RawSummary.model_validate(json.loads(text[start : end + 1]))


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def evidence_in_abstract(evidence: str, abstract: str) -> bool:
    """Whitespace-normalized, case-insensitive substring check. Deliberately literal: no
    quote/dash folding and no fuzzy matching, so 'verified' means 'really in the abstract'."""
    e = _norm(evidence)
    return len(e) >= MIN_EVIDENCE_CHARS and e in _norm(abstract)


def verify_summary(raw: RawSummary, abstract: str) -> PaperSummary:
    """Keep only claims whose evidence is quoted from the abstract; count the rest."""
    out: dict = {"not_stated": [], "unverified_claims": 0}
    for name in CLAIM_FIELDS:
        claims = getattr(raw, name)
        if claims == NOT_STATED:
            out["not_stated"].append(name)
            continue
        kept = [c for c in claims if evidence_in_abstract(c.evidence, abstract)]
        out["unverified_claims"] += len(claims) - len(kept)
        out[name] = kept
    return PaperSummary(**out)


class LLMSummarizer:
    """Structured summary whose claims are each backed by a verified quote from the abstract.

    Grounding is against the abstract only; the full text is never read."""

    def __init__(self, client: LLMClient, workers: int = 4):
        self.client = client
        self.workers = workers
        self.id = f"llm:{client.model}:{PROMPT_VERSION}"

    def _ask(self, abstract: str) -> RawSummary:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Abstract:\n{abstract}"},
        ]
        for attempt in range(MAX_RETRIES + 1):
            reply = self.client.complete(messages)
            try:
                return parse_summary_json(reply)
            except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
                if attempt == MAX_RETRIES:
                    raise SummarizeError(f"no valid JSON after {attempt + 1} attempts: {e}") from e
                log.warning("invalid LLM summary JSON (attempt %d): %s", attempt + 1, e)
                messages += [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": f"That reply was invalid: {e}\nReply again with only the "
                        "corrected JSON object.",
                    },
                ]
        raise AssertionError("unreachable")

    def summarize(self, paper: PaperIn) -> PaperSummary:
        try:
            raw = self._ask(paper.abstract)
        except LLMError as e:
            raise SummarizeError(str(e)) from e
        return verify_summary(raw, paper.abstract)


def summarize_all(
    summarizer: Summarizer, papers: list[PaperIn]
) -> tuple[list[PaperSummary], list[str]]:
    """Summarize every paper (in parallel if the backend allows) and return the summaries in
    order. A paper that fails gets a placeholder and its error is returned for `warnings`."""
    errors: list[str] = []

    def run(p: PaperIn) -> PaperSummary:
        try:
            return summarizer.summarize(p)
        except SummarizeError as e:
            log.warning("summarizing %r failed: %s", p.title, e)
            errors.append(f"summary failed for {p.title!r}: {e}")
            return PaperSummary(overview="Summary unavailable.")

    workers = max(1, min(getattr(summarizer, "workers", 1), len(papers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, papers)), errors

import pytest

from app.config import Settings
from app.services.llm import make_llm_client
from app.services.summarize import LLMSummarizer, evidence_in_abstract
from tests.test_summarize import ABSTRACT, PAPER


@pytest.mark.llm
def test_real_llm_returns_grounded_claims():
    """Needs --run-llm and a configured endpoint (LLM_MODEL plus LLM_API_KEY and/or
    LLM_BASE_URL, or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic). Makes real, billable calls."""
    client = make_llm_client(Settings())
    if client is None:
        pytest.skip("no LLM configured in the environment")
    s = LLMSummarizer(client).summarize(PAPER)  # no cache: always a real call
    kept = [c for name in ("problem", "method", "results", "limitations") for c in getattr(s, name)]
    assert kept or s.unverified_claims  # the model produced something
    assert all(evidence_in_abstract(c.evidence, ABSTRACT) for c in kept)

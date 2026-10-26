from app.services.llm import LLMCache
from app.services.summarize import PROMPT_VERSION, LLMSummarizer
from tests.conftest import ScriptedLLM
from tests.test_summarize import GOOD, OTHER, PAPER


def test_cache_roundtrip_persists_across_instances(tmp_path):
    path = tmp_path / "sub" / "c.db"
    LLMCache(path).put("k", "v")
    assert LLMCache(path).get("k") == "v"
    assert LLMCache(path).get("missing") is None


def test_put_overwrites(tmp_path):
    cache = LLMCache(tmp_path / "c.db")
    cache.put("k", "old")
    cache.put("k", "new")
    assert cache.get("k") == "new"


def test_key_depends_on_model_version_and_text():
    keys = {
        LLMCache.key("m", "v1", "abstract"),
        LLMCache.key("m2", "v1", "abstract"),
        LLMCache.key("m", "v2", "abstract"),
        LLMCache.key("m", "v1", "abstract2"),
    }
    assert len(keys) == 4
    assert LLMCache.key("m", "v1", "abstract") == LLMCache.key("m", "v1", "abstract")
    assert LLMCache.key("a", "bc", "d") != LLMCache.key("ab", "c", "d")  # no concatenation clash


def test_second_call_is_a_cache_hit_and_skips_the_llm(tmp_path):
    llm = ScriptedLLM(GOOD)
    summarizer = LLMSummarizer(llm, LLMCache(tmp_path / "c.db"))
    first = summarizer.summarize(PAPER)
    second = LLMSummarizer(llm, LLMCache(tmp_path / "c.db")).summarize(PAPER)  # new instance
    assert len(llm.calls) == 1
    assert first == second


def test_different_abstract_or_model_misses(tmp_path):
    cache = LLMCache(tmp_path / "c.db")
    llm = ScriptedLLM(GOOD)
    LLMSummarizer(llm, cache).summarize(PAPER)
    LLMSummarizer(llm, cache).summarize(OTHER)
    other_model = ScriptedLLM(GOOD)
    other_model.model = "another-model"
    LLMSummarizer(other_model, cache).summarize(PAPER)
    assert (len(llm.calls), len(other_model.calls)) == (2, 1)


def test_invalid_replies_are_not_cached(tmp_path):
    cache = LLMCache(tmp_path / "c.db")
    llm = ScriptedLLM("garbage")
    try:
        LLMSummarizer(llm, cache).summarize(PAPER)
    except Exception:  # SummarizeError, covered in test_summarize
        pass
    assert cache.get(cache.key("fake-llm", PROMPT_VERSION, PAPER.abstract)) is None


def test_unreadable_cache_entry_is_recomputed_and_verification_reruns(tmp_path):
    cache = LLMCache(tmp_path / "c.db")
    key = cache.key("fake-llm", PROMPT_VERSION, PAPER.abstract)
    cache.put(key, "not json")
    llm = ScriptedLLM(GOOD)
    s = LLMSummarizer(llm, cache).summarize(PAPER)
    assert len(llm.calls) == 1 and s.problem
    assert cache.get(key) != "not json"  # healed

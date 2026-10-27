import pytest

from app.main import app, get_summarizers, get_theme_writer
from app.schemas import Claim, Paper, PaperIn, PaperSummary
from app.services.cluster import keyword_labels
from app.services.llm import LLMCache, LLMError
from app.services.summarize import LLMSummarizer
from app.services.theme import SYSTEM_PROMPT, ThemeWriter, check_citations
from tests.conftest import ScriptedLLM
from tests.test_summarize import ABSTRACT, GOOD


def paper(title, claims=("Sparse attention saves memory.",), **kw):
    structured = (
        PaperSummary(problem=[Claim(text=t, evidence="e") for t in claims]) if claims else None
    )
    return Paper(title=title, summary="s", link="l", structured=structured, **kw)


# ---- keyword labels ----


def test_labels_pick_distinguishing_terms():
    labels = keyword_labels(
        {
            "0": ["Protein folding with deep networks.", "Protein structure prediction models."],
            "1": ["Robot grasping with tactile sensors.", "Robot manipulation using sensors."],
        }
    )
    assert "protein" in labels["0"] and "robot" in labels["1"]
    assert "protein" not in labels["1"] and "robot" not in labels["0"]


def test_terms_shared_by_every_cluster_do_not_dominate():
    shared = "learning " * 2
    labels = keyword_labels({"0": [shared + "protein"], "1": [shared + "robot"]})
    assert labels["0"].split(", ")[0] == "protein"
    assert labels["1"].split(", ")[0] == "robot"


def test_noise_cluster_gets_fixed_label_and_is_not_scored():
    labels = keyword_labels({"0": ["protein folding"], "-1": ["zzz outlier text"]})
    assert labels["-1"] == "Misc / unclustered"
    assert "zzz" not in labels["0"]
    assert keyword_labels({"-1": ["only noise"]}) == {"-1": "Misc / unclustered"}


def test_labels_fall_back_when_no_usable_words():
    assert keyword_labels({"0": ["the of and"], "1": ["a an it"]}) == {
        "0": "Cluster 1",
        "1": "Cluster 2",
    }
    assert keyword_labels({}) == {}


def test_labels_are_deterministic_and_limited_to_top_n():
    docs = {"0": ["alpha beta gamma delta epsilon"], "1": ["omega"]}
    assert keyword_labels(docs) == keyword_labels(docs)
    assert len(keyword_labels(docs, top_n=2)["0"].split(", ")) == 2


# ---- citation checking ----


def test_valid_citations_are_untouched():
    assert check_citations("A [1]. B [2][3].", 3) == ("A [1]. B [2][3].", [], 3)


def test_out_of_range_citations_are_stripped_and_reported():
    text, invalid, valid = check_citations("A [1]. B [7] and C [0].", 2)
    assert text == "A [1]. B and C."
    assert invalid == [7, 0]
    assert valid == 1


def test_mixed_brackets_keep_the_valid_numbers():
    text, invalid, valid = check_citations("Both [1, 9] agree.", 2)
    assert (text, invalid, valid) == ("Both [1] agree.", [9], 1)


def test_non_citation_brackets_are_left_alone():
    assert check_citations("See [note] and [1].", 1)[0] == "See [note] and [1]."


# ---- ThemeWriter ----


def cluster_of(n=3):
    return [paper(f"P{i}", claims=(f"claim {i}",)) for i in range(n)]


def test_theme_is_written_from_verified_claims_only():
    llm = ScriptedLLM("They share a focus on memory [1][2]. One adds speed [3].")
    theme = ThemeWriter(llm).write(cluster_of())
    assert theme.text == "They share a focus on memory [1][2]. One adds speed [3]."
    assert theme.invalid_citations == []
    messages = llm.calls[0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[2] P1" in messages[1]["content"] and "problem: claim 2" in messages[1]["content"]


def test_out_of_range_citation_is_flagged_and_stripped():
    theme = ThemeWriter(ScriptedLLM("Memory matters [1]. Speed too [5].")).write(cluster_of(2))
    assert theme.text == "Memory matters [1]. Speed too."
    assert theme.invalid_citations == [5]


def test_theme_without_any_valid_citation_is_rejected():
    theme = ThemeWriter(ScriptedLLM("Everything is great [9].")).write(cluster_of(2))
    assert theme.text is None and theme.invalid_citations == [9]
    assert ThemeWriter(ScriptedLLM("No citations at all.")).write(cluster_of(2)).text is None


def test_nothing_to_synthesize_makes_no_llm_call():
    llm = ScriptedLLM("x [1]")
    writer = ThemeWriter(llm)
    assert writer.write(cluster_of(1)) is None  # a single paper
    assert writer.write([paper("a", claims=()), paper("b", claims=())]) is None  # no claims
    assert llm.calls == []


def test_theme_cache_hit_skips_the_llm(tmp_path):
    cache = LLMCache(tmp_path / "c.db")
    llm = ScriptedLLM("Cached theme [1].")
    ThemeWriter(llm, cache).write(cluster_of(2))
    again = ThemeWriter(llm, cache).write(cluster_of(2))
    assert again.text == "Cached theme [1]." and len(llm.calls) == 1
    ThemeWriter(llm, cache).write(cluster_of(3))  # different claims -> different key
    assert len(llm.calls) == 2


# ---- through the API ----


@pytest.fixture
def themed(client, summarizer, fetcher):
    summ_llm = ScriptedLLM(GOOD)
    theme_llm = ScriptedLLM("The papers study sparse attention [1][2]. Bogus [9].")
    app.dependency_overrides[get_summarizers] = lambda: {
        "hf": summarizer,
        "llm": LLMSummarizer(summ_llm),
    }
    app.dependency_overrides[get_theme_writer] = lambda: ThemeWriter(theme_llm)
    fetcher.papers = [
        PaperIn(title=f"Sparse Attention {i}", abstract=ABSTRACT, link=f"http://x/{i}")
        for i in range(2)
    ]
    return client, theme_llm


def test_review_includes_labels_and_cited_theme(themed):
    client, _ = themed
    r = client.get("/review", params={"topic": "t", "summarizer": "llm", "num_clusters": 1}).json()
    info = r["cluster_info"]["0"]
    assert info["label"]
    assert info["theme_summary"] == "The papers study sparse attention [1][2]. Bogus."
    assert info["invalid_citations"] == [9]
    again = client.get(f"/reviews/{r['review_id']}").json()
    assert again["cluster_info"] == r["cluster_info"]  # persisted


def test_hf_backend_gets_labels_but_no_theme(themed):
    client, theme_llm = themed
    r = client.get("/review", params={"topic": "t", "num_clusters": 1}).json()
    assert r["cluster_info"]["0"]["label"] and r["cluster_info"]["0"]["theme_summary"] is None
    assert theme_llm.calls == []


def test_theme_failure_is_a_warning_not_an_error(themed):
    client, theme_llm = themed

    def boom(messages):
        raise LLMError("rate limited")

    theme_llm.complete = boom
    r = client.get("/review", params={"topic": "t", "summarizer": "llm", "num_clusters": 1}).json()
    assert r["cluster_info"]["0"]["theme_summary"] is None
    assert any("rate limited" in w for w in r["warnings"])


def test_rejected_theme_warns(themed):
    client, theme_llm = themed
    theme_llm.replies = ["Nothing cited."]
    r = client.get("/review", params={"topic": "t", "summarizer": "llm", "num_clusters": 1}).json()
    assert r["cluster_info"]["0"]["theme_summary"] is None
    assert any("no valid citations" in w for w in r["warnings"])


def test_hdbscan_noise_group_is_labelled_and_gets_no_theme(client, fetcher, monkeypatch):
    from app.services import cluster

    monkeypatch.setitem(
        cluster.CLUSTERERS,
        "hdbscan",
        type(
            "Fake", (), {"name": "hdbscan", "cluster": lambda self, v, k: [0, 0, -1, -1, -1, -1]}
        )(),
    )
    r = client.get("/review", params={"topic": "t", "cluster_method": "hdbscan"}).json()
    assert list(r["clusters"]) == ["0", "-1"]
    assert r["cluster_info"]["-1"] == {
        "label": "Misc / unclustered",
        "theme_summary": None,
        "invalid_citations": [],
    }

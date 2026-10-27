import csv

from eval import build_benchmark as bb
from eval.common import cached, key_hash
from tests.conftest import FakeEmbedder

LONG = "x" * 400


def rec(i, cat="cs.CL", date="2023-01-02", abstract=LONG):
    return {
        "id": f"2301.{i:05d}v1",
        "title": f"T{i}",
        "abstract": abstract,
        "category": cat,
        "published": date,
    }


class FakeSearch:
    def __init__(self, records):
        self.records, self.calls = records, 0

    def __call__(self, query, n):
        self.calls += 1
        return self.records


def test_fetch_filters_crosslists_stubs_and_duplicates_then_sorts_and_trims(tmp_path):
    records = [
        rec(3, date="2023-01-09"),
        rec(1, date="2023-01-02"),
        rec(2, cat="cs.LG"),  # primary category differs: cross-listed into cs.CL
        rec(4, abstract="too short"),
        rec(1, date="2023-01-02"),  # duplicate id
        rec(5, date="2023-01-05"),
    ]
    got = bb.fetch_category("cs.CL", 2, FakeSearch(records), tmp_path)
    assert [p["id"] for p in got] == ["2301.00001", "2301.00005"]  # earliest first, versions cut


def test_fetch_is_cached(tmp_path):
    search = FakeSearch([rec(1)])
    bb.fetch_category("cs.CL", 1, search, tmp_path)
    bb.fetch_category("cs.CL", 1, search, tmp_path)
    assert search.calls == 1


def test_part_a_records_shortfalls(tmp_path):
    def search(query, n):
        cat = query.split()[0].removeprefix("cat:")
        return [rec(i, cat=cat) for i in range(2 if cat == "cs.DB" else 5)]

    data = bb.build_part_a(search, per=3, cache_dir=tmp_path)
    assert data["meta"]["shortfalls"] == {"cs.DB": 2}
    assert data["meta"]["n_papers"] == 5 * 3 + 2
    assert {p["category"] for p in data["part_a"]} == set(bb.CATEGORIES)


def test_handlabel_template_has_empty_labels_and_is_never_overwritten(tmp_path):
    def search(topic, n):
        return [
            {
                "id": f"{abs(hash(topic)) % 1000}.{i}",
                "title": f"{topic} paper {i}",
                "abstract": f"abstract about {topic} number {i}",
            }
            for i in range(8)
        ]

    path = tmp_path / "t.csv"
    kwargs = dict(path=path, topics=["alpha", "beta"], per_topic=5, cache_dir=tmp_path / "c")
    assert bb.build_handlabel_template(FakeEmbedder(), search, **kwargs) == "written"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 10 and {r["topic"] for r in rows} == {"alpha", "beta"}
    assert all(r["cluster_label"] == "" for r in rows)
    path.write_text("my labels", encoding="utf-8")
    assert bb.build_handlabel_template(FakeEmbedder(), search, **kwargs) == "exists"
    assert path.read_text(encoding="utf-8") == "my labels"


def test_cached_helper_roundtrip_and_key_stability(tmp_path):
    calls = []
    for _ in range(2):
        cached("ns", {"a": 1, "b": [2]}, lambda: calls.append(1) or {"v": 1}, tmp_path)
    assert len(calls) == 1
    assert key_hash({"a": 1, "b": 2}) == key_hash({"b": 2, "a": 1})
    assert cached("ns", {"a": 2}, lambda: {"v": 2}, tmp_path) == {"v": 2}

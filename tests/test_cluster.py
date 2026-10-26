import numpy as np
import pytest

from app.services.cluster import (
    CLUSTERERS,
    NOISE,
    AgglomerativeClusterer,
    HDBSCANClusterer,
    KMeansClusterer,
)
from tests.conftest import FakeEmbedder, make_papers


def vectors(n):
    return FakeEmbedder().encode([f"text {i}" for i in range(n)])


def blobs(per_blob=10, centers=3, dims=6, outliers=0, seed=0):
    """Tight, well separated blobs along different axes, plus lone outliers on spare axes."""
    rng = np.random.default_rng(seed)
    pts, truth = [], []
    for c in range(centers):
        pts += list(np.eye(dims)[c] + rng.normal(0, 0.03, (per_blob, dims)))
        truth += [c] * per_blob
    for o in range(outliers):
        pts.append(np.eye(dims)[centers + o] + rng.normal(0, 0.03, dims))
        truth.append(NOISE)
    return np.array(pts), truth


def same_partition(labels, truth):
    """True when both labelings group the points identically (label names may differ)."""
    pairs = set(zip(labels, truth, strict=True))
    return len({a for a, _ in pairs}) == len({b for _, b in pairs}) == len(pairs)


@pytest.mark.parametrize("cls", [KMeansClusterer, AgglomerativeClusterer])
@pytest.mark.parametrize("n,k,expected", [(0, 3, 0), (1, 3, 1), (2, 3, 2), (10, 3, 3), (3, 8, 3)])
def test_fixed_k_never_crashes_and_clamps(cls, n, k, expected):
    labels = cls().cluster(vectors(n), k)
    assert len(labels) == n
    assert len(set(labels)) == expected


@pytest.mark.parametrize("cls", [KMeansClusterer, AgglomerativeClusterer])
def test_k_one_gives_a_single_cluster(cls):
    assert cls().cluster(vectors(3), 1) == [0, 0, 0]


@pytest.mark.parametrize("cls", [KMeansClusterer, AgglomerativeClusterer])
def test_fixed_k_recovers_blobs(cls):
    x, truth = blobs()
    assert same_partition(cls().cluster(x, 3), truth)


@pytest.mark.parametrize("cls", [KMeansClusterer, AgglomerativeClusterer])
@pytest.mark.parametrize("centers", [2, 3, 4])
def test_auto_k_finds_the_number_of_blobs(cls, centers):
    x, truth = blobs(centers=centers)
    labels = cls().cluster(x, None)
    assert len(set(labels)) == centers
    assert same_partition(labels, truth)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_auto_k_with_too_few_points_is_one_cluster(n):
    assert KMeansClusterer().cluster(vectors(n), None) == [0] * n


def test_hdbscan_finds_blobs_and_ignores_k():
    x, truth = blobs()
    labels = HDBSCANClusterer().cluster(x, 99)
    assert NOISE not in labels
    assert same_partition(labels, truth)


def test_hdbscan_marks_outliers_as_noise():
    x, truth = blobs(outliers=3)
    labels = HDBSCANClusterer().cluster(x, None)
    assert [lab for lab, t in zip(labels, truth, strict=True) if t == NOISE] == [NOISE] * 3
    assert same_partition(labels, truth)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_hdbscan_tiny_inputs(n):
    assert HDBSCANClusterer().cluster(vectors(n), None) == [0] * n


def test_registry_has_all_methods():
    assert set(CLUSTERERS) == {"kmeans", "agglomerative", "hdbscan"}
    assert all(c.name == name for name, c in CLUSTERERS.items())


# ---- through the API ----


def test_unknown_cluster_method_is_422(client):
    r = client.get("/review", params={"topic": "t", "cluster_method": "dbscan"})
    assert r.status_code == 422


@pytest.mark.parametrize("bad", ["0", "9", "abc", ""])
def test_bad_num_clusters_is_422(client, bad):
    assert client.get("/review", params={"topic": "t", "num_clusters": bad}).status_code == 422


def test_auto_num_clusters_and_each_method_work(client, fetcher):
    fetcher.papers = make_papers(8)
    for method in CLUSTERERS:
        r = client.get(
            "/review", params={"topic": "t", "cluster_method": method, "num_clusters": "auto"}
        )
        assert r.status_code == 200, method
        assert sum(len(v) for v in r.json()["clusters"].values()) == 8


def test_method_and_k_are_part_of_the_cache_key(client, fetcher):
    fetcher.papers = make_papers(6)
    base = {"topic": "t"}
    assert client.get("/review", params=base).json()["cached"] is False
    assert (
        client.get("/review", params=base | {"cluster_method": "agglomerative"}).json()["cached"]
        is False
    )
    assert client.get("/review", params=base | {"num_clusters": "auto"}).json()["cached"] is False
    assert client.get("/review", params=base).json()["cached"] is True


def test_hdbscan_ignores_k_in_the_cache_key(client, fetcher):
    fetcher.papers = make_papers(6)
    p = {"topic": "t", "cluster_method": "hdbscan"}
    client.get("/review", params=p | {"num_clusters": 2})
    assert client.get("/review", params=p | {"num_clusters": 5}).json()["cached"] is True

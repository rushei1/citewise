import pytest

from app.services.cluster import Clusterer
from tests.conftest import FakeEmbedder


@pytest.mark.parametrize("n,k,expected_clusters", [(0, 3, 0), (1, 3, 1), (2, 3, 2), (10, 3, 3)])
def test_cluster_counts(n, k, expected_clusters):
    labels = Clusterer(FakeEmbedder()).cluster([f"text {i}" for i in range(n)], k)
    assert len(labels) == n
    assert len(set(labels)) == expected_clusters


def test_num_clusters_larger_than_papers():
    labels = Clusterer(FakeEmbedder()).cluster(["a", "b", "c"], 8)
    assert sorted(set(labels)) == [0, 1, 2]


def test_single_cluster_requested():
    assert Clusterer(FakeEmbedder()).cluster(["a", "b", "c"], 1) == [0, 0, 0]

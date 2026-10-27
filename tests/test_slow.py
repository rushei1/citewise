import pytest

from app.schemas import PaperIn
from app.services.cluster import CLUSTERERS, Embedder
from app.services.summarize import HFSummarizer

ABSTRACTS = [
    (
        "Attention Is All You Need",
        "The dominant sequence transduction models are based on complex recurrent or "
        "convolutional neural networks. We propose the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely.",
    ),
    (
        "Deep Residual Learning for Image Recognition",
        "Deeper neural networks are more difficult to train. We present a residual learning "
        "framework to ease the training of networks that are substantially deeper than those "
        "used previously, reformulating layers as learning residual functions.",
    ),
    (
        "Playing Atari with Deep Reinforcement Learning",
        "We present the first deep learning model to successfully learn control policies "
        "directly from high-dimensional sensory input using reinforcement learning.",
    ),
]


@pytest.mark.slow
def test_real_models_end_to_end():
    summarizer = HFSummarizer()
    summaries = [
        summarizer.summarize(PaperIn(title=t, abstract=a, link="l")).overview for t, a in ABSTRACTS
    ]
    assert all(len(s) > 10 for s in summaries)
    labels = CLUSTERERS["kmeans"].cluster(Embedder().encode(summaries), 2)
    assert len(labels) == 3

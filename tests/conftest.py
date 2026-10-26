import pytest

MARKERS = ("slow", "llm", "network")


def pytest_addoption(parser):
    for name in MARKERS:
        parser.addoption(
            f"--run-{name}", action="store_true", default=False, help=f"run @{name} tests"
        )


def pytest_collection_modifyitems(config, items):
    for item in items:
        for name in MARKERS:
            if name in item.keywords and not config.getoption(f"--run-{name}"):
                item.add_marker(pytest.mark.skip(reason=f"needs --run-{name}"))

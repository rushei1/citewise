import pytest
import requests

from app.services.http import Http, HttpError, RateLimiter


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = {"ok": True} if body is None else body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class Script:
    """Replays a list of responses/exceptions as the `get` callable; records calls."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def make_http(*steps, **kw):
    sleeps = []
    script = Script(*steps)
    http = Http(get=script, sleep=sleeps.append, rand=lambda: 0.0, **kw)
    return http, script, sleeps


def test_success_passes_timeout_and_headers():
    http, script, sleeps = make_http(FakeResponse(body={"a": 1}), timeout=7)
    assert http.get_json("http://x", {"q": 1}, {"h": "v"}) == {"a": 1}
    assert script.calls[0] == {
        "url": "http://x",
        "params": {"q": 1},
        "headers": {"h": "v"},
        "timeout": 7,
    }
    assert sleeps == []


def test_429_honors_retry_after_then_succeeds():
    http, script, sleeps = make_http(
        FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(body={"ok": 1})
    )
    assert http.get_json("http://x") == {"ok": 1}
    assert sleeps == [3.0]
    assert len(script.calls) == 2


def test_retry_after_is_capped():
    http, _, sleeps = make_http(
        FakeResponse(429, headers={"Retry-After": "999"}), FakeResponse(), max_delay=10
    )
    http.get_json("http://x")
    assert sleeps == [10]


def test_exponential_backoff_without_retry_after():
    http, script, sleeps = make_http(
        FakeResponse(503), FakeResponse(502), FakeResponse(500), FakeResponse(), retries=4
    )
    http.get_json("http://x")
    assert sleeps == [1.0, 2.0, 4.0]  # base * 2**n, jitter pinned to 0
    assert len(script.calls) == 4


def test_jitter_is_added_and_backoff_capped():
    sleeps = []
    script = Script(FakeResponse(503), FakeResponse(503), FakeResponse())
    http = Http(get=script, sleep=sleeps.append, rand=lambda: 0.5, base_delay=2.0, max_delay=5.0)
    http.get_json("http://x")
    assert sleeps == [2.0 + 1.0, 5.0]  # 2*1 + .5*2, then min(cap, 4 + 1)


def test_gives_up_after_retries():
    http, script, sleeps = make_http(*[FakeResponse(429)] * 3, retries=3)
    with pytest.raises(HttpError, match="gave up after 3 attempts"):
        http.get_json("http://x")
    assert len(script.calls) == 3
    assert len(sleeps) == 2  # no sleep after the final attempt


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(status):
    http, script, sleeps = make_http(FakeResponse(status))
    with pytest.raises(HttpError, match=str(status)):
        http.get_json("http://x")
    assert len(script.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("exc", [requests.ConnectionError("down"), requests.Timeout("slow")])
def test_network_errors_are_retried(exc):
    http, script, _ = make_http(exc, FakeResponse())
    assert http.get_json("http://x") == {"ok": True}
    assert len(script.calls) == 2


def test_other_request_exceptions_fail_fast():
    http, script, _ = make_http(requests.exceptions.InvalidURL("bad"))
    with pytest.raises(HttpError):
        http.get_json("http://x")
    assert len(script.calls) == 1


def test_invalid_json_fails_fast():
    http, script, _ = make_http(FakeResponse(body=ValueError("nope")))
    with pytest.raises(HttpError, match="invalid JSON"):
        http.get_json("http://x")
    assert len(script.calls) == 1


def test_non_numeric_retry_after_falls_back_to_backoff():
    http, _, sleeps = make_http(
        FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), FakeResponse()
    )
    http.get_json("http://x")
    assert sleeps == [1.0]


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_rate_limiter_spaces_calls():
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock.time, sleep=clock.sleep)
    limiter.wait()  # first call goes straight through
    limiter.wait()
    limiter.wait()
    assert clock.slept == [1.0, 1.0]


def test_rate_limiter_no_sleep_when_enough_time_passed():
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock.time, sleep=clock.sleep)
    limiter.wait()
    clock.now += 5
    limiter.wait()
    assert clock.slept == []


def test_http_uses_limiter_on_every_attempt():
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock.time, sleep=clock.sleep)
    http, script, _ = make_http(FakeResponse(503), FakeResponse(), limiter=limiter)
    http.get_json("http://x")
    assert len(script.calls) == 2
    assert clock.slept == [2.0]

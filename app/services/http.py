import logging
import random
import threading
import time
from collections.abc import Callable

import requests
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class HttpError(Exception):
    """An external HTTP call failed for good (non-retryable, or retries exhausted)."""


class _Retryable(HttpError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class RateLimiter:
    """Allow at most one call per `min_interval` seconds (per source)."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._next_ok = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_ok - now)
            self._next_ok = max(now, self._next_ok) + self.min_interval
        if delay:
            self._sleep(delay)


def _retry_after(res: requests.Response) -> float | None:
    """Seconds from a numeric Retry-After header (HTTP-date form is ignored)."""
    try:
        return max(0.0, float(res.headers.get("Retry-After", "")))
    except ValueError:
        return None


class Http:
    """GET-JSON with timeout, exponential backoff + jitter, Retry-After and optional rate limit.

    `get`, `sleep` and `rand` are injectable so tests need no network and no real waiting."""

    def __init__(
        self,
        timeout: float = 15.0,
        retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        limiter: RateLimiter | None = None,
        get: Callable[..., requests.Response] = requests.get,
        post: Callable[..., requests.Response] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
    ):
        self.timeout = timeout
        self.retries = retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.limiter = limiter
        self._get = get
        self._post = post
        self._sleep = sleep
        self._rand = rand

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception()
        if isinstance(exc, _Retryable) and exc.retry_after is not None:
            return min(exc.retry_after, self.max_delay)
        backoff = self.base_delay * 2 ** (state.attempt_number - 1)
        return min(self.max_delay, backoff + self._rand() * self.base_delay)

    def _attempt(self, send: Callable[[], requests.Response], url: str):
        if self.limiter:
            self.limiter.wait()
        try:
            res = send()
        except (requests.ConnectionError, requests.Timeout) as e:
            raise _Retryable(f"{type(e).__name__}: {e}") from e
        except requests.RequestException as e:
            raise HttpError(str(e)) from e
        if res.status_code in RETRY_STATUSES:
            raise _Retryable(f"HTTP {res.status_code} from {url}", _retry_after(res))
        if res.status_code != 200:
            raise HttpError(f"HTTP {res.status_code} from {url}")
        try:
            return res.json()
        except ValueError as e:
            raise HttpError(f"invalid JSON from {url}") from e

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        return self._request(
            lambda: self._get(url, params=params, headers=headers, timeout=self.timeout), url
        )

    def post_json(self, url: str, body: dict, headers: dict | None = None):
        return self._request(
            lambda: self._post(url, json=body, headers=headers, timeout=self.timeout), url
        )

    def _request(self, send: Callable[[], requests.Response], url: str):
        retrying = Retrying(
            stop=stop_after_attempt(self.retries),
            wait=self._wait,
            retry=retry_if_exception(lambda e: isinstance(e, _Retryable)),
            sleep=self._sleep,
            before_sleep=lambda s: log.warning(
                "retrying %s (attempt %d failed: %s)", url, s.attempt_number, s.outcome.exception()
            ),
            reraise=True,
        )
        try:
            return retrying(self._attempt, send, url)
        except _Retryable as e:
            raise HttpError(f"gave up after {self.retries} attempts: {e}") from e

import pytest

from app.config import Settings
from app.services.http import Http, HttpError
from app.services.llm import AnthropicClient, LLMError, OpenAICompatClient, make_llm_client
from tests.test_http import FakeResponse

MSGS = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]


class FakeHttp:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []

    def post_json(self, url, body, headers=None):
        self.calls.append((url, body, headers))
        if self.error:
            raise self.error
        return self.payload


def test_openai_request_shape_and_usage():
    http = FakeHttp(
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
    )
    client = OpenAICompatClient("m", "http://localhost:11434/v1/", api_key="k", http=http)
    assert client.complete(MSGS) == "hello"
    url, body, headers = http.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert body == {"model": "m", "messages": MSGS, "temperature": 0}
    assert headers == {"Authorization": "Bearer k"}
    assert (client.usage.prompt_tokens, client.usage.completion_tokens) == (5, 2)


def test_openai_without_key_sends_no_auth_header():
    http = FakeHttp({"choices": [{"message": {"content": "x"}}]})
    OpenAICompatClient("m", http=http).complete(MSGS)
    assert http.calls[0][2] is None


def test_openai_errors_become_llm_error():
    with pytest.raises(LLMError, match="request failed"):
        OpenAICompatClient("m", http=FakeHttp(error=HttpError("HTTP 401"))).complete(MSGS)
    with pytest.raises(LLMError, match="unexpected"):
        OpenAICompatClient("m", http=FakeHttp({"nope": 1})).complete(MSGS)


def test_anthropic_splits_system_prompt():
    http = FakeHttp(
        {
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    )
    client = AnthropicClient("claude-x", "key", http=http)
    assert client.complete(MSGS) == "ab"
    _, body, headers = http.calls[0]
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert headers["x-api-key"] == "key"
    assert client.usage.completion_tokens == 1
    with pytest.raises(LLMError):
        AnthropicClient("m", "k", http=FakeHttp({"oops": 1})).complete(MSGS)
    with pytest.raises(LLMError):
        AnthropicClient("m", "k", http=FakeHttp(error=HttpError("x"))).complete(MSGS)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, None),
        ({"llm_api_key": "k"}, None),  # no model
        ({"llm_model": "m"}, None),  # no key and no base URL
        ({"llm_model": "m", "llm_api_key": "k"}, OpenAICompatClient),
        ({"llm_model": "m", "llm_base_url": "http://localhost:11434/v1"}, OpenAICompatClient),
        ({"llm_provider": "anthropic", "llm_model": "m"}, None),
        (
            {"llm_provider": "anthropic", "llm_model": "m", "anthropic_api_key": "k"},
            AnthropicClient,
        ),
    ],
)
def test_make_llm_client(env, expected):
    client = make_llm_client(Settings(_env_file=None, **env))
    assert (type(client) is expected) if expected else client is None


def test_post_json_retries_like_get():
    sleeps, calls = [], []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return (
            FakeResponse(429, headers={"Retry-After": "2"}) if len(calls) == 1 else FakeResponse()
        )

    http = Http(post=post, sleep=sleeps.append)
    assert http.post_json("http://x", {"a": 1}) == {"ok": True}
    assert calls == [{"a": 1}, {"a": 1}]
    assert sleeps == [2.0]

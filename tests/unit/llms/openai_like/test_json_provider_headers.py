"""
Tests for the `headers` field of JSON-configured providers (providers.json).

An entry's `headers` are sent on chat completions (streamed and not, sync and async),
the Responses API and /v1/messages, both when the model serves those endpoints natively
(`supported_endpoints` / `model_supported_endpoints`) and when they are bridged through
chat completions. `{api_key}` in a value is replaced by the resolved API key. A header the
caller passes with the same name (any casing) wins, and `Authorization` is never taken
from an entry. Embeddings do not get entry headers.
"""

import json
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.openai_like.dynamic_config import create_responses_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams

_BASE: Final = {"base_url": "https://headers.example.com/v1", "api_key_env": "HEADERS_EXAMPLE_API_KEY"}
_RESPONSE: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "m",
    "status": "completed",
    "output": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
}
_MESSAGE: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_NATIVE_ONLY: Final = {"model_supported_endpoints": {"native-m": ["/v1/responses", "/v1/messages"]}}
_COMPLETION: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
_STREAM: Final = (
    "data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }
    )
    + "\n\ndata: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    )
    + "\n\ndata: [DONE]\n\n"
)


def _provider(slug: str, **overrides) -> SimpleProviderConfig:
    return SimpleProviderConfig(slug=slug, data={**_BASE, **overrides})


@pytest.fixture
def registered(monkeypatch):
    """Registers a JSON provider for one test, as providers.json would"""

    def register(slug: str, **overrides) -> SimpleProviderConfig:
        provider = _provider(slug, **overrides)
        monkeypatch.setitem(JSONProviderRegistry._providers, slug, provider)
        return provider

    return register


class TestLoader:
    def test_headers_are_read_from_the_entry(self):
        provider = _provider("read_headers", headers={"X-Client": "litellm"})
        assert dict(provider.headers) == {"X-Client": "litellm"}

    def test_entry_without_headers_has_none(self):
        assert dict(_provider("no_headers").headers) == {}
        assert _provider("no_headers").request_headers("sk-1") == {}

    @pytest.mark.parametrize("headers", [{"X-Count": 1}, {"X-Flag": True}, ["X-Client", "litellm"], "X-Client"])
    def test_non_string_headers_are_rejected(self, headers):
        with pytest.raises(ValidationError):
            _provider("bad_headers", headers=headers)

    def test_api_key_placeholder_is_replaced(self):
        provider = _provider("placeholder", headers={"api-subscription-key": "{api_key}", "X-Client": "litellm"})
        assert provider.request_headers("sk-123") == {"api-subscription-key": "sk-123", "X-Client": "litellm"}

    def test_placeholder_header_is_left_out_without_an_api_key(self):
        provider = _provider("placeholder_no_key", headers={"api-subscription-key": "{api_key}", "X-Client": "litellm"})
        assert provider.request_headers(None) == {"X-Client": "litellm"}

    @pytest.mark.parametrize("name", ["Authorization", "authorization", "AUTHORIZATION"])
    def test_authorization_is_never_taken_from_an_entry(self, name):
        provider = _provider("no_auth_override", headers={name: "Bearer entry", "X-Client": "litellm"})
        assert provider.request_headers("sk-1") == {"X-Client": "litellm"}

    def test_registry_returns_nothing_for_unknown_or_missing_providers(self):
        assert JSONProviderRegistry.get_request_headers("not-a-json-provider", "sk-1") == {}
        assert JSONProviderRegistry.get_request_headers(None, "sk-1") == {}

    def test_sarvam_entry_headers_are_now_active(self):
        sarvam = JSONProviderRegistry.get("sarvam")
        assert sarvam is not None
        assert sarvam.request_headers("sk-sarvam") == {"api-subscription-key": "sk-sarvam"}


class TestChatCompletions:
    @pytest.mark.parametrize("stream", [False, True])
    def test_entry_headers_are_sent(self, registered, stream):
        registered("hdr_chat", headers={"X-Client": "litellm"})
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_COMPLETION)
                if not stream
                else httpx.Response(200, headers={"content-type": "text/event-stream"}, text=_STREAM)
            )
            response = litellm.completion(
                model="hdr_chat/m", messages=[{"role": "user", "content": "hi"}], api_key="sk-chat", stream=stream
            )
            if stream:
                for _ in response:
                    pass

        request = route.calls.last.request
        assert request.headers["x-client"] == "litellm"
        assert request.headers["authorization"] == "Bearer sk-chat"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_entry_headers_are_sent_async(self, registered, monkeypatch, stream):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        registered("hdr_chat_async", headers={"X-Client": "litellm"})
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_COMPLETION)
                if not stream
                else httpx.Response(200, headers={"content-type": "text/event-stream"}, text=_STREAM)
            )
            response = await litellm.acompletion(
                model="hdr_chat_async/m", messages=[{"role": "user", "content": "hi"}], api_key="sk-chat", stream=stream
            )
            if stream:
                async for _ in response:
                    pass

        assert route.calls.last.request.headers["x-client"] == "litellm"

    def test_caller_header_with_the_same_name_wins(self, registered):
        registered("hdr_caller_wins", headers={"X-Client": "litellm"})
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").respond(200, json=_COMPLETION)
            litellm.completion(
                model="hdr_caller_wins/m",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-chat",
                extra_headers={"x-client": "my-app"},
            )

        assert route.calls.last.request.headers.get_list("x-client") == ["my-app"]

    def test_api_key_placeholder_uses_the_env_key(self, registered, monkeypatch):
        registered("hdr_env_key", headers={"api-subscription-key": "{api_key}"})
        monkeypatch.setenv("HEADERS_EXAMPLE_API_KEY", "sk-from-env")
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").respond(200, json=_COMPLETION)
            litellm.completion(model="hdr_env_key/m", messages=[{"role": "user", "content": "hi"}])

        assert route.calls.last.request.headers["api-subscription-key"] == "sk-from-env"

    def test_provider_without_headers_sends_no_extra_header(self, registered):
        registered("hdr_none")
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").respond(200, json=_COMPLETION)
            litellm.completion(model="hdr_none/m", messages=[{"role": "user", "content": "hi"}], api_key="sk-chat")

        assert "x-client" not in route.calls.last.request.headers
        assert "api-subscription-key" not in route.calls.last.request.headers


class TestResponsesApi:
    def test_entry_headers_are_added(self):
        config = create_responses_config_class(_provider("hdr_responses", headers={"X-Client": "litellm"}))()
        headers = config.validate_environment(
            headers={}, model="m", litellm_params=GenericLiteLLMParams(api_key="sk-r")
        )
        assert headers == {"Authorization": "Bearer sk-r", "X-Client": "litellm"}

    def test_caller_header_wins(self):
        config = create_responses_config_class(_provider("hdr_responses_caller", headers={"X-Client": "litellm"}))()
        headers = config.validate_environment(
            headers={"x-client": "my-app"}, model="m", litellm_params=GenericLiteLLMParams(api_key="sk-r")
        )
        assert headers["x-client"] == "my-app"
        assert "X-Client" not in headers

    def test_caller_extra_header_wins_before_it_is_merged(self):
        # The Responses handler merges the caller's extra_headers after validate_environment;
        # the entry header must not be added next to a caller header that differs only in case.
        config = create_responses_config_class(_provider("hdr_responses_extra", headers={"X-Client": "litellm"}))()
        headers = config.validate_environment(
            headers={},
            model="m",
            litellm_params=GenericLiteLLMParams(api_key="sk-r", extra_headers={"x-client": "my-app"}),
        )
        assert "X-Client" not in headers


class TestAnthropicMessages:
    def test_entry_headers_are_added(self):
        config = JSONProviderAnthropicMessagesConfig(_provider("hdr_messages", headers={"X-Client": "litellm"}))
        headers, _ = config.validate_anthropic_messages_environment(
            headers={}, model="m", messages=[], optional_params={}, litellm_params={}, api_key="sk-m"
        )
        assert headers["X-Client"] == "litellm"
        assert headers["authorization"] == "Bearer sk-m"

    def test_caller_header_wins(self):
        config = JSONProviderAnthropicMessagesConfig(_provider("hdr_messages_caller", headers={"X-Client": "litellm"}))
        headers, _ = config.validate_anthropic_messages_environment(
            headers={"x-client": "my-app"},
            model="m",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-m",
        )
        assert headers["x-client"] == "my-app"
        assert "X-Client" not in headers


class TestNativeAndBridgedPaths:
    """
    A provider whose `native-m` model serves /v1/responses and /v1/messages natively
    (`model_supported_endpoints`), while its other models reach them through the chat
    completions bridge. The entry header must be sent exactly once on every path, and a
    caller header with the same name must win on every path.
    """

    @pytest.mark.parametrize("caller", [None, "my-app"])
    def test_responses_native_model(self, registered, caller):
        registered("hdr_paths_rn", headers={"X-Client": "litellm"}, **_NATIVE_ONLY)
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/responses").respond(200, json=_RESPONSE)
            litellm.responses(
                model="hdr_paths_rn/native-m",
                input="hi",
                api_key="sk-r",
                **({"extra_headers": {"x-client": caller}} if caller else {}),
            )

        assert route.call_count == 1
        assert route.calls.last.request.headers.get_list("x-client") == [caller or "litellm"]

    @pytest.mark.parametrize("caller", [None, "my-app"])
    def test_responses_bridged_model(self, registered, caller):
        registered("hdr_paths_rb", headers={"X-Client": "litellm"}, **_NATIVE_ONLY)
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").respond(200, json=_COMPLETION)
            litellm.responses(
                model="hdr_paths_rb/other-m",
                input="hi",
                api_key="sk-r",
                **({"extra_headers": {"x-client": caller}} if caller else {}),
            )

        assert route.call_count == 1
        assert route.calls.last.request.headers.get_list("x-client") == [caller or "litellm"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("caller", [None, "my-app"])
    async def test_messages_native_model(self, registered, monkeypatch, caller):
        # Native /v1/messages is only chosen for providers listed in LlmProviders, so this case
        # re-registers an existing one (prism) with the test entry for the duration of the test.
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
        registered("prism", headers={"X-Client": "litellm"}, **_NATIVE_ONLY)
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/messages").respond(200, json=_MESSAGE)
            await litellm.anthropic.messages.acreate(
                model="prism/native-m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                api_key="sk-m",
                **({"extra_headers": {"x-client": caller}} if caller else {}),
            )

        assert route.call_count == 1
        assert route.calls.last.request.headers.get_list("x-client") == [caller or "litellm"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("caller", [None, "my-app"])
    async def test_messages_bridged_model(self, registered, monkeypatch, caller):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        registered("hdr_paths_mb", headers={"X-Client": "litellm"}, **_NATIVE_ONLY)
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/chat/completions").respond(200, json=_COMPLETION)
            await litellm.anthropic.messages.acreate(
                model="hdr_paths_mb/other-m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                api_key="sk-m",
                **({"extra_headers": {"x-client": caller}} if caller else {}),
            )

        assert route.call_count == 1
        assert route.calls.last.request.headers.get_list("x-client") == [caller or "litellm"]

    def test_embeddings_get_no_entry_headers(self, registered):
        registered("hdr_paths_emb", headers={"X-Client": "litellm"}, supported_endpoints=["/v1/embeddings"])
        with respx.mock() as upstream:
            route = upstream.post("https://headers.example.com/v1/embeddings").respond(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                    "model": "e",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
            litellm.embedding(model="hdr_paths_emb/e", input=["hi"], api_key="sk-e")

        assert route.call_count == 1
        assert "x-client" not in route.calls.last.request.headers


class TestGreenferenceEntry:
    def test_entry(self):
        greenference = JSONProviderRegistry.get("greenference")
        assert greenference is not None
        assert greenference.base_url == "https://api.greenference.com/v1"
        assert greenference.api_key_env == "GREENFERENCE_API_KEY"
        assert greenference.api_base_env == "GREENFERENCE_API_BASE"
        assert greenference.base_class == "openai_gpt"
        assert dict(greenference.headers) == {"X-Greenference-Client": "litellm"}

    def test_provider_resolution(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("GREENFERENCE_API_KEY", "gf-test-key")
        monkeypatch.delenv("GREENFERENCE_API_BASE", raising=False)
        model, provider, api_key, api_base = get_llm_provider(model="greenference/qwen3-32b")

        assert (model, provider, api_key, api_base) == (
            "qwen3-32b",
            "greenference",
            "gf-test-key",
            "https://api.greenference.com/v1",
        )

    def test_completion_sends_the_client_header(self, monkeypatch):
        monkeypatch.setenv("GREENFERENCE_API_KEY", "gf-test-key")
        monkeypatch.delenv("GREENFERENCE_API_BASE", raising=False)
        with respx.mock() as upstream:
            route = upstream.post("https://api.greenference.com/v1/chat/completions").respond(200, json=_COMPLETION)
            litellm.completion(model="greenference/qwen3-32b", messages=[{"role": "user", "content": "hi"}])

        request = route.calls.last.request
        assert request.headers["x-greenference-client"] == "litellm"
        assert request.headers["authorization"] == "Bearer gf-test-key"
        assert json.loads(request.content)["model"] == "qwen3-32b"

    def test_responses_goes_through_the_bridge_and_sends_the_client_header(self, monkeypatch):
        # The entry declares no supported_endpoints, so /v1/responses is bridged to chat completions.
        monkeypatch.setenv("GREENFERENCE_API_KEY", "gf-test-key")
        monkeypatch.delenv("GREENFERENCE_API_BASE", raising=False)
        with respx.mock() as upstream:
            route = upstream.post("https://api.greenference.com/v1/chat/completions").respond(200, json=_COMPLETION)
            litellm.responses(model="greenference/qwen3-32b", input="hi")

        assert route.call_count == 1
        assert route.calls.last.request.headers.get_list("x-greenference-client") == ["litellm"]

"""
Tests for the OpenCode bridge model layer (app/model.py).

These cover the contract between SourceCheck AI and bridge/opencode_bridge.py:
factory selection, default wiring, and every failure mode the bridge can return —
including the free-tier quota response, which must not be reported as a generic
500. The bridge itself is stubbed with httpx.MockTransport, so no listener and no
network are required.
"""

import json

import httpx
import pytest

from app.config import settings
from app.model import (
    ModelError,
    ModelTimeoutError,
    OpenAICompatibleLLM,
    OpenCodeBridgeLLM,
    get_model,
)

BRIDGE_URL = "http://127.0.0.1:18383/v1"


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_retry_by_default(monkeypatch):
    """Keep existing tests instant: without this, every 429/502 case would sleep
    through the full backoff schedule (2s + 4s + 8s) before failing."""
    monkeypatch.setattr(settings, "BRIDGE_MAX_RETRIES", 0)


def make_llm(handler, **kwargs) -> OpenCodeBridgeLLM:
    """Build an OpenCodeBridgeLLM whose HTTP calls are served by `handler`."""
    kwargs.setdefault("endpoint", BRIDGE_URL)
    kwargs.setdefault("model_name", "opencode/mimo-v2.6-flash-free")
    return OpenCodeBridgeLLM(transport=httpx.MockTransport(handler), **kwargs)


def ok_handler(text="hello"):
    def _handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    return _handle


def error_handler(status: int, message: str):
    def _handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message}})

    return _handle


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------


class TestModelFactory:
    def test_opencode_provider_returns_bridge_llm(self):
        assert isinstance(get_model("opencode"), OpenCodeBridgeLLM)

    @pytest.mark.parametrize("alias", ["opencode_bridge", "bridge"])
    def test_aliases_map_to_bridge_llm(self, alias):
        assert isinstance(get_model(alias), OpenCodeBridgeLLM)

    def test_bridge_llm_is_an_openai_compatible_llm(self):
        # Reusing the chat-completions wire protocol is deliberate; assert the
        # inheritance so a future refactor cannot silently break that assumption.
        assert issubclass(OpenCodeBridgeLLM, OpenAICompatibleLLM)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError) as exc:
            get_model("not-a-provider")
        assert "not-a-provider" in str(exc.value)

    def test_mock_provider_still_selected_by_default(self, monkeypatch):
        # Default must stay 'mock' so the suite never needs a live bridge.
        from app.config import settings

        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
        from app.model import MockLLM

        assert isinstance(get_model(), MockLLM)


# ------------------------------------------------------------------------------
# Defaults / wiring
# ------------------------------------------------------------------------------


class TestBridgeConfiguration:
    def test_default_endpoint_model_and_timeout(self, monkeypatch):
        # The factory reads settings at call time, so an ambient BRIDGE_ENDPOINT
        # (docker-compose points the web service at http://bridge:18383/v1) would
        # otherwise mask the documented defaults this test exists to pin down.
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", BRIDGE_URL)
        monkeypatch.setattr(settings, "BRIDGE_MODEL", "opencode/mimo-v2.6-flash-free")
        llm = OpenCodeBridgeLLM(transport=httpx.MockTransport(ok_handler()))
        assert llm.endpoint == BRIDGE_URL
        assert llm.model_name == "opencode/mimo-v2.6-flash-free"
        assert llm.timeout > 0

    def test_endpoint_is_normalised_for_trailing_slash(self):
        llm = OpenCodeBridgeLLM(endpoint="http://127.0.0.1:18383/v1/",
                                transport=httpx.MockTransport(ok_handler()))
        assert llm.endpoint == "http://127.0.0.1:18383/v1"

    def test_no_auth_header_by_default(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        make_llm(handler).generate("hi")
        assert seen["auth"] is None  # bridge needs no key unless one is configured

    def test_api_key_is_sent_when_configured(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        make_llm(handler, api_key="s3cret").generate("hi")
        assert seen["auth"] == "Bearer s3cret"


# ------------------------------------------------------------------------------
# generate()
# ------------------------------------------------------------------------------


class TestGenerate:
    def test_returns_message_content(self):
        assert make_llm(ok_handler("bridge ok")).generate("hi") == "bridge ok"

    def test_sends_model_and_roles(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        make_llm(handler).generate("the prompt", system_prompt="be terse")
        body = captured["body"]
        assert body["model"] == "opencode/mimo-v2.6-flash-free"
        assert body["messages"][0] == {"role": "system", "content": "be terse"}
        assert body["messages"][1] == {"role": "user", "content": "the prompt"}

    def test_system_prompt_omitted_when_empty(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        make_llm(handler).generate("hi")
        assert [m["role"] for m in captured["body"]["messages"]] == ["user"]


class TestGenerateFailures:
    def test_unreachable_bridge_names_the_fix(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(ModelError) as exc:
            make_llm(handler).generate("hi")
        message = str(exc.value)
        assert BRIDGE_URL in message
        # The message must tell the operator how to start the backend.
        assert "run-bridge.ps1" in message

    def test_timeout_maps_to_model_timeout_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(ModelTimeoutError):
            make_llm(handler, timeout=1.0).generate("hi")

    def test_timeout_message_suggests_the_setting(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(ModelTimeoutError) as exc:
            make_llm(handler, timeout=1.0).generate("hi")
        assert "BRIDGE_TIMEOUT_SECONDS" in str(exc.value)

    def test_400_surfaces_bridge_message_and_models_hint(self):
        # Mirror the bridge's real wording so the "no duplication" check is meaningful.
        bridge_text = (
            "model 'gpt-4o' is not available on this bridge. It serves free OpenCode "
            "models: opencode/mimo-v2.6-flash-free. See GET /v1/models."
        )
        with pytest.raises(ModelError) as exc:
            make_llm(error_handler(400, bridge_text)).generate("hi")
        message = str(exc.value)
        assert "not available" in message
        # The hint URL must be exactly base + /models. Building it as "/v1/models"
        # produced http://...:18383/v1/v1/models because the endpoint already has /v1.
        assert f"{BRIDGE_URL}/models" in message
        assert "/v1/v1/" not in message
        # One from the bridge's own text, one from our hint — and nothing extra.
        assert message.count("/models") == 2

    def test_429_quota_is_reported_as_rate_limiting(self):
        with pytest.raises(ModelError) as exc:
            make_llm(error_handler(429, "Free usage limit reached. Retry later.")).generate("hi")
        message = str(exc.value)
        assert "Free usage limit reached" in message
        assert "rate-limiting" in message or "busy" in message

    def test_unexpected_status_includes_code_and_body(self):
        with pytest.raises(ModelError) as exc:
            make_llm(error_handler(502, "upstream exploded")).generate("hi")
        message = str(exc.value)
        assert "502" in message
        assert "upstream exploded" in message

    def test_non_json_body_does_not_raise_value_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        with pytest.raises(ModelError) as exc:
            make_llm(handler).generate("hi")
        assert "unexpected payload" in str(exc.value)

    def test_error_body_without_json_falls_back_to_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="plain text failure")

        with pytest.raises(ModelError) as exc:
            make_llm(handler).generate("hi")
        assert "plain text failure" in str(exc.value)


# ------------------------------------------------------------------------------
# generate_answer()
# ------------------------------------------------------------------------------


class TestGenerateAnswer:
    PASSAGE_TEXT = "The Eiffel Tower was completed on 31 March 1889."

    @staticmethod
    def _passages():
        from app.schemas import EvidencePassage

        return [
            EvidencePassage(
                text=TestGenerateAnswer.PASSAGE_TEXT,
                source_url="https://example.com/eiffel",
                page_title="Eiffel Tower",
                score=0.9,
            )
        ]

    def test_parses_json_response(self):
        payload = {
            "answer": "31 March 1889",
            "insufficient_evidence": False,
            "claims": [{"claim": "completed 31 March 1889",
                        "url": "https://example.com/eiffel", "quote": self.PASSAGE_TEXT}],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

        out = make_llm(handler).generate_answer("When?", self._passages())
        assert out["answer"] == "31 March 1889"
        assert out["insufficient_evidence"] is False
        assert out["claims"][0]["url"] == "https://example.com/eiffel"

    def test_extracts_json_wrapped_in_prose(self):
        # Free models frequently preface the object with a sentence.
        content = 'Here is the result:\n{"answer": "31 March 1889", "insufficient_evidence": true, "claims": []}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        out = make_llm(handler).generate_answer("When?", self._passages())
        assert out["answer"] == "31 March 1889"
        assert out["insufficient_evidence"] is True

    def test_non_json_output_falls_back_to_raw_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "just plain prose"}}]})

        out = make_llm(handler).generate_answer("When?", self._passages())
        assert out["answer"] == "just plain prose"
        # Fallback must still cite the top passage rather than fabricate a URL.
        assert out["claims"][0]["url"] == "https://example.com/eiffel"

    def test_empty_passages_still_returns_valid_shape(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "no json here"}}]})

        out = make_llm(handler).generate_answer("When?", [])
        assert out["claims"] == []

    def test_prompt_wraps_evidence_in_untrusted_block(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        make_llm(handler).generate_answer("When?", self._passages())
        user_prompt = captured["body"]["messages"][-1]["content"]
        assert "<untrusted_evidence>" in user_prompt
        assert self.PASSAGE_TEXT in user_prompt
        assert "https://example.com/eiffel" in user_prompt

    def test_failures_propagate_as_model_errors(self):
        with pytest.raises(ModelError):
            make_llm(error_handler(429, "Free usage limit reached")).generate_answer("When?", self._passages())


# ------------------------------------------------------------------------------
# Retry / backoff behaviour
# ------------------------------------------------------------------------------


class TestRetryBehaviour:
    """The bridge can transiently 429 (free-tier burst limits). The agent should
    ride those out, but must NOT sleep through a multi-hour quota lockout."""

    @staticmethod
    def sequence(*responses):
        """Serve each response in order, then keep returning the last one."""
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            return item

        return handler

    @pytest.fixture
    def sleep_calls(self, monkeypatch):
        slept = []
        monkeypatch.setattr("app.model.time.sleep", slept.append)
        return slept

    @pytest.fixture
    def retries_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "BRIDGE_MAX_RETRIES", 3)
        monkeypatch.setattr(settings, "BRIDGE_RETRY_BACKOFF_SECONDS", 2.0)

    def test_recovers_after_transient_429(self, retries_enabled, sleep_calls):
        responses = [
            httpx.Response(429, json={"error": {"message": "bridge busy"}}),
            httpx.Response(429, json={"error": {"message": "bridge busy"}}),
            httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]}),
        ]
        llm = make_llm(self.sequence(*responses))
        assert llm.generate("hi") == "recovered"
        # Exponential: 2.0, then 4.0.
        assert sleep_calls == [2.0, 4.0]

    def test_gives_up_after_exhausting_retries(self, retries_enabled, sleep_calls):
        responses = [httpx.Response(429, json={"error": {"message": "bridge busy"}})] * 5
        llm = make_llm(self.sequence(*responses))
        with pytest.raises(ModelError) as exc:
            llm.generate("hi")
        assert "rate-limiting" in str(exc.value)
        # 3 retries => sleeps before attempts 2, 3, 4.
        assert len(sleep_calls) == 3

    def test_does_not_retry_a_long_quota_lockout(self, retries_enabled, sleep_calls):
        # The bridge appends "(retry in 17h)" for a hard free-tier lockout. Sleeping
        # three times cannot fix that - fail fast and surface the hint instead.
        responses = [httpx.Response(
            429, json={"error": {"message": "Free usage limit reached. (retry in 17h)"}})]
        llm = make_llm(self.sequence(*responses))
        with pytest.raises(ModelError) as exc:
            llm.generate("hi")
        assert "retry in 17h" in str(exc.value)
        assert sleep_calls == []  # not a single pointless sleep

    def test_retries_a_short_declared_wait(self, retries_enabled, sleep_calls):
        # A wait inside BRIDGE_MAX_RETRY_WAIT_SECONDS (300s) is worth riding out.
        responses = [
            httpx.Response(429, json={"error": {"message": "busy (retry in 30s)"}}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
        llm = make_llm(self.sequence(*responses))
        assert llm.generate("hi") == "ok"
        assert sleep_calls == [2.0]

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_retries_server_errors(self, status, retries_enabled, sleep_calls):
        responses = [
            httpx.Response(status, json={"error": {"message": "upstream exploded"}}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
        llm = make_llm(self.sequence(*responses))
        assert llm.generate("hi") == "ok"
        assert len(sleep_calls) == 1

    def test_never_retries_a_400(self, retries_enabled, sleep_calls):
        # A bad model id will never succeed on retry.
        responses = [httpx.Response(400, json={"error": {"message": "model not available"}})] * 5
        llm = make_llm(self.sequence(*responses))
        with pytest.raises(ModelError):
            llm.generate("hi")
        assert sleep_calls == []

    def test_connection_error_fails_fast_without_retrying(self, retries_enabled, sleep_calls):
        # "bridge not running" is actionable immediately - no point sleeping first.
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        llm = OpenCodeBridgeLLM(transport=httpx.MockTransport(handler))
        with pytest.raises(ModelError) as exc:
            llm.generate("hi")
        assert "run-bridge.ps1" in str(exc.value)
        assert sleep_calls == []

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("plain message", None),
            ("busy (retry in 30s)", 30),
            ("busy (retry in 5m)", 300),
            ("busy (retry in 17h)", 61200),
            ("retry in 12", None),          # no unit -> not a hint we understand
        ],
    )
    def test_parses_retry_hint(self, message, expected):
        assert OpenCodeBridgeLLM._parse_retry_hint_seconds(message) == expected

import hashlib
import hmac
import json

import pytest

from gateway.platforms.omnideck_translation import (
    SIGNATURE_HEADER,
    _build_prompt,
    _endpoint_config,
    _parse_translation_response,
    _preflight_codex_usage_rotation,
    _validated_payload,
    _verify_signature,
    handle_omnideck_translation_request,
)


class _FakeRequest:
    def __init__(self, body: bytes, headers=None):
        self._body = body
        self.headers = headers or {}
        self.content_length = len(body)

    async def read(self):
        return self._body


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def test_verify_omnideck_signature_plain_and_prefixed():
    body = b'{"field_values":{"title":"Dress"},"locale":"ja"}'
    secret = "test-secret"
    sig = _sign(body, secret)

    assert _verify_signature(body, sig, secret)
    assert _verify_signature(body, f"sha256={sig}", secret)
    assert not _verify_signature(body, "deadbeef", secret)


def test_validated_payload_requires_field_values():
    with pytest.raises(Exception):
        _validated_payload({"locale": "ja", "field_values": {}})


def test_prompt_includes_correction_context():
    payload = _validated_payload(
        {
            "locale": "de",
            "field_values": {"title": "Cuffed Pocket Cargo Shorts"},
            "correction_context": "untranslated_english: translate into German",
        }
    )

    prompt = _build_prompt(payload)

    assert "Mandatory validator feedback" in prompt
    assert "untranslated_english: translate into German" in prompt


def test_parse_translation_response_extracts_json_from_text():
    output = 'Here is the result:\n{"translations":{"title":"ワンピース"}}\n'
    assert _parse_translation_response(output, {"title"}) == {"title": "ワンピース"}


def test_endpoint_config_defaults_codex_usage_rotation_threshold():
    config = _endpoint_config({})
    assert config["codex_rotate_used_percent"] == 90.0


def test_preflight_codex_usage_rotation_rotates_when_threshold_exceeded(monkeypatch):
    class Window:
        label = "Weekly"
        used_percent = 95.0

    class Snapshot:
        windows = (Window(),)

    class Pool:
        def __init__(self):
            self.called = False

        def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
            self.called = True
            assert status_code == 429
            assert error_context["reason"] == "near_usage_limit"
            return type("Entry", (), {"id": "next-cred"})()

    pool = Pool()
    monkeypatch.setattr(
        "agent.account_usage._fetch_codex_account_usage",
        lambda: Snapshot(),
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda provider: pool,
    )

    assert _preflight_codex_usage_rotation(used_percent_threshold=90.0) is True
    assert pool.called is True


def test_preflight_codex_usage_rotation_skips_below_threshold(monkeypatch):
    class Window:
        label = "Weekly"
        used_percent = 50.0

    class Snapshot:
        windows = (Window(),)

    monkeypatch.setattr(
        "agent.account_usage._fetch_codex_account_usage",
        lambda: Snapshot(),
    )

    assert _preflight_codex_usage_rotation(used_percent_threshold=90.0) is False


@pytest.mark.asyncio
async def test_handle_omnideck_translation_request_returns_translations(monkeypatch):
    secret = "test-secret"

    async def fake_run(prompt, **kwargs):
        assert "Source fields JSON" in prompt
        return json.dumps({"translations": {"title": "ワンピース"}})

    monkeypatch.setattr(
        "gateway.platforms.omnideck_translation._run_hermes_oneshot",
        fake_run,
    )

    body = json.dumps(
        {
            "shop_id": 30001,
            "resource_type": "PRODUCT",
            "resource_id": "gid://shopify/Product/1",
            "locale": "ja",
            "field_values": {"title": "Dress"},
        }
    ).encode("utf-8")
    request = _FakeRequest(
        body,
        headers={SIGNATURE_HEADER: _sign(body, secret)},
    )
    resp = await handle_omnideck_translation_request(
        request,
        config_extra={
            "omnideck_translation": {
                "secret": secret,
                "hermes_bin": "/bin/echo",
            }
        },
        max_body_bytes=1_000_000,
    )
    assert resp.status == 200
    assert json.loads(resp.text) == {"translations": {"title": "ワンピース"}}


@pytest.mark.asyncio
async def test_handle_omnideck_translation_request_rejects_bad_signature():
    body = json.dumps({"locale": "ja", "field_values": {"title": "Dress"}}).encode(
        "utf-8"
    )
    request = _FakeRequest(body, headers={SIGNATURE_HEADER: "bad"})
    resp = await handle_omnideck_translation_request(
        request,
        config_extra={"omnideck_translation": {"secret": "test-secret"}},
        max_body_bytes=1_000_000,
    )
    assert resp.status == 401

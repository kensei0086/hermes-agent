"""Omnideck synchronous translation endpoint for the webhook gateway.

This is intentionally separate from the generic webhook subscription flow:
Omnideck waits for a JSON response, so the handler must translate during the
HTTP request and return ``{"translations": {...}}`` directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - imported only when gateway is enabled
    web = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 180
MAX_FIELDS = 30
MAX_FIELD_CHARS = 30_000
MAX_TOTAL_FIELD_CHARS = 100_000
SIGNATURE_HEADER = "X-Omnideck-Hermes-Signature"


class OmnideckTranslationError(Exception):
    """HTTP-safe endpoint error."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _endpoint_config(extra: Mapping[str, Any]) -> Dict[str, Any]:
    raw = extra.get("omnideck_translation", {})
    if not isinstance(raw, Mapping):
        raw = {}

    secret = (
        str(raw.get("secret") or "").strip()
        or os.getenv("HERMES_OMNIDECK_TRANSLATION_SECRET", "").strip()
        or os.getenv("HERMES_DISPATCH_HTTP_SECRET", "").strip()
    )
    venv_hermes = Path(sys.executable).with_name("hermes")
    hermes_bin = (
        str(raw.get("hermes_bin") or "").strip()
        or os.getenv("HERMES_OMNIDECK_TRANSLATION_HERMES_BIN", "").strip()
        or shutil.which("hermes")
        or (str(venv_hermes) if venv_hermes.exists() else "")
        or ""
    )
    timeout = raw.get("timeout_seconds") or os.getenv(
        "HERMES_OMNIDECK_TRANSLATION_TIMEOUT_SECONDS", ""
    )
    try:
        timeout_seconds = int(timeout)
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    rotate_percent = raw.get("codex_rotate_used_percent") or os.getenv(
        "HERMES_OMNIDECK_CODEX_ROTATE_USED_PERCENT",
        "90",
    )
    try:
        codex_rotate_used_percent = float(rotate_percent)
    except (TypeError, ValueError):
        codex_rotate_used_percent = 90.0

    return {
        "enabled": _truthy(raw.get("enabled", True)),
        "secret": secret,
        "hermes_bin": hermes_bin,
        "timeout_seconds": max(10, min(timeout_seconds, 600)),
        "model": str(raw.get("model") or os.getenv("HERMES_OMNIDECK_TRANSLATION_MODEL", "")).strip(),
        "provider": str(raw.get("provider") or os.getenv("HERMES_OMNIDECK_TRANSLATION_PROVIDER", "")).strip(),
        "codex_rotate_used_percent": max(0.0, min(codex_rotate_used_percent, 100.0)),
    }


def _verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    if not secret:
        return False
    received = (signature or "").strip()
    if received.startswith("sha256="):
        received = received[len("sha256=") :]
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received, expected)


def _validated_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise OmnideckTranslationError(400, "invalid_payload", "JSON object required")

    locale = str(payload.get("locale") or "").strip()
    if not locale:
        raise OmnideckTranslationError(400, "missing_locale", "locale is required")

    field_values = payload.get("field_values")
    if not isinstance(field_values, dict) or not field_values:
        raise OmnideckTranslationError(
            400,
            "missing_field_values",
            "field_values must be a non-empty object",
        )
    if len(field_values) > MAX_FIELDS:
        raise OmnideckTranslationError(413, "too_many_fields", "too many fields")

    normalized_fields: Dict[str, str] = {}
    total_chars = 0
    for key, value in field_values.items():
        field_key = str(key or "").strip()
        if not field_key:
            raise OmnideckTranslationError(400, "invalid_field", "field key is empty")
        if not isinstance(value, str):
            raise OmnideckTranslationError(
                400,
                "invalid_field_value",
                f"field {field_key} must be a string",
            )
        if len(value) > MAX_FIELD_CHARS:
            raise OmnideckTranslationError(
                413,
                "field_too_large",
                f"field {field_key} is too large",
            )
        total_chars += len(value)
        if total_chars > MAX_TOTAL_FIELD_CHARS:
            raise OmnideckTranslationError(
                413,
                "payload_too_large",
                "translation fields are too large",
            )
        normalized_fields[field_key] = value

    return {
        "shop_id": payload.get("shop_id"),
        "resource_type": str(payload.get("resource_type") or "").strip() or "PRODUCT",
        "resource_id": str(payload.get("resource_id") or "").strip(),
        "locale": locale,
        "field_values": normalized_fields,
        "correction_context": str(payload.get("correction_context") or "").strip(),
    }


def _build_prompt(payload: Mapping[str, Any]) -> str:
    fields = json.dumps(payload["field_values"], ensure_ascii=False, separators=(",", ":"))
    context = {
        "shop_id": payload.get("shop_id"),
        "resource_type": payload.get("resource_type"),
        "resource_id": payload.get("resource_id"),
        "locale": payload.get("locale"),
    }
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    correction_context = str(payload.get("correction_context") or "").strip()
    correction_section = ""
    if correction_context:
        correction_section = (
            "\nMandatory validator feedback for this attempt:\n"
            f"{correction_context}\n"
        )

    return (
        "You are the Omnideck production translation engine.\n"
        "Translate the provided Shopify product fields into the target locale.\n"
        "Return ONLY strict JSON in this exact shape: "
        '{"translations":{"field_name":"translated text"}}\n'
        "Rules:\n"
        "- Translate every input field key exactly once.\n"
        "- Preserve HTML tags, placeholders, numbers, SKUs, brand names, URLs, and JSON-like snippets.\n"
        "- Do not add explanations, markdown, code fences, comments, or extra top-level keys.\n"
        "- If a value is empty, return an empty string for that same field.\n"
        "- Do not invent product facts.\n\n"
        f"Context: {context_json}\n"
        f"Target locale: {payload['locale']}\n"
        f"{correction_section}"
        f"Source fields JSON: {fields}\n"
    )


def _parse_translation_response(output: str, expected_keys: set[str]) -> Dict[str, str]:
    text = (output or "").strip()
    if not text:
        raise OmnideckTranslationError(502, "empty_hermes_response", "Hermes returned no text")

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise OmnideckTranslationError(
                502,
                "invalid_hermes_json",
                "Hermes response did not contain a JSON object",
            )
        try:
            decoded = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OmnideckTranslationError(
                502,
                "invalid_hermes_json",
                "Hermes response JSON could not be parsed",
            ) from exc

    translations = decoded.get("translations") if isinstance(decoded, dict) else None
    if not isinstance(translations, dict):
        raise OmnideckTranslationError(
            502,
            "missing_translations",
            "Hermes response is missing translations object",
        )

    normalized: Dict[str, str] = {}
    missing = []
    for key in expected_keys:
        value = translations.get(key)
        if value is None:
            missing.append(key)
            continue
        normalized[key] = str(value)

    if missing:
        raise OmnideckTranslationError(
            502,
            "missing_translation_fields",
            "Hermes response omitted required translation fields",
        )
    return normalized


async def _run_hermes_oneshot(
    prompt: str,
    *,
    hermes_bin: str,
    timeout_seconds: int,
    model: str = "",
    provider: str = "",
) -> str:
    if not hermes_bin:
        raise OmnideckTranslationError(
            503,
            "hermes_binary_missing",
            "Hermes CLI binary is not configured",
        )

    cmd = [hermes_bin, "--ignore-rules"]
    if model:
        cmd.extend(["--model", model])
    if provider:
        cmd.extend(["--provider", provider])
    cmd.extend(["--oneshot", prompt])

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        try:
            process.kill()  # type: ignore[has-type]
        except Exception:
            pass
        raise OmnideckTranslationError(
            504,
            "hermes_timeout",
            "Hermes translation timed out",
        ) from exc
    except FileNotFoundError as exc:
        raise OmnideckTranslationError(
            503,
            "hermes_binary_missing",
            "Hermes CLI binary was not found",
        ) from exc

    if process.returncode != 0:
        logger.warning(
            "[omnideck-translation] Hermes oneshot failed returncode=%s stderr_len=%d",
            process.returncode,
            len(stderr or b""),
        )
        raise OmnideckTranslationError(
            502,
            "hermes_failed",
            "Hermes translation process failed",
        )

    return stdout.decode("utf-8", errors="replace")


def _preflight_codex_usage_rotation(*, used_percent_threshold: float) -> bool:
    """Rotate openai-codex credentials before a subscription window is exhausted.

    The main Hermes agent already rotates after 402/429 failures. This preflight
    covers production translation, where waiting for a hard failure can delay the
    customer-facing pipeline. If the usage endpoint is unavailable, translation
    should continue and rely on the runtime error recovery path.
    """
    if used_percent_threshold <= 0:
        return False

    try:
        from agent.account_usage import _fetch_codex_account_usage
        from agent.credential_pool import load_pool
    except Exception as exc:
        logger.debug("[omnideck-translation] codex usage preflight unavailable: %s", exc)
        return False

    try:
        snapshot = _fetch_codex_account_usage()
    except Exception as exc:
        logger.warning("[omnideck-translation] codex usage preflight failed: %s", exc)
        return False

    if not snapshot or not snapshot.windows:
        return False

    hot_windows = [
        window
        for window in snapshot.windows
        if window.used_percent is not None and float(window.used_percent) >= used_percent_threshold
    ]
    if not hot_windows:
        return False

    reason = ", ".join(
        f"{window.label}:{float(window.used_percent):.1f}%"
        for window in hot_windows
    )
    try:
        pool = load_pool("openai-codex")
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=429,
            error_context={
                "reason": "near_usage_limit",
                "message": f"Codex usage above {used_percent_threshold:.1f}% ({reason})",
            },
        )
    except Exception as exc:
        logger.warning("[omnideck-translation] codex preflight rotation failed: %s", exc)
        return False

    if next_entry is None:
        logger.warning(
            "[omnideck-translation] codex usage above threshold but no alternate credential exists (%s)",
            reason,
        )
        return False

    logger.warning(
        "[omnideck-translation] codex usage above threshold; rotated to credential id=%s (%s)",
        getattr(next_entry, "id", "?"),
        reason,
    )
    return True


async def handle_omnideck_translation_request(
    request: "web.Request",
    *,
    config_extra: Mapping[str, Any],
    max_body_bytes: int,
) -> "web.Response":
    config = _endpoint_config(config_extra)
    if not config["enabled"]:
        return web.json_response(
            {"error": "omnideck_translation_disabled"},
            status=503,
        )
    if not config["secret"]:
        return web.json_response(
            {"error": "omnideck_translation_secret_missing"},
            status=503,
        )

    content_length = request.content_length or 0
    if content_length > max_body_bytes:
        return web.json_response({"error": "payload_too_large"}, status=413)

    try:
        raw_body = await request.read()
    except Exception:
        return web.json_response({"error": "bad_request"}, status=400)

    if not _verify_signature(
        raw_body,
        request.headers.get(SIGNATURE_HEADER, ""),
        config["secret"],
    ):
        logger.warning("[omnideck-translation] invalid signature")
        return web.json_response({"error": "invalid_signature"}, status=401)

    try:
        payload = _validated_payload(json.loads(raw_body))
        _preflight_codex_usage_rotation(
            used_percent_threshold=float(config["codex_rotate_used_percent"]),
        )
        prompt = _build_prompt(payload)
        output = await _run_hermes_oneshot(
            prompt,
            hermes_bin=config["hermes_bin"],
            timeout_seconds=config["timeout_seconds"],
            model=config["model"],
            provider=config["provider"],
        )
        translations = _parse_translation_response(
            output,
            set(payload["field_values"].keys()),
        )
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_json"}, status=400)
    except OmnideckTranslationError as exc:
        return web.json_response(
            {"error": exc.code, "message": exc.message},
            status=exc.status,
        )
    except Exception:
        logger.exception("[omnideck-translation] unexpected failure")
        return web.json_response({"error": "translation_failed"}, status=500)

    logger.info(
        "[omnideck-translation] translated resource_type=%s locale=%s fields=%d",
        payload.get("resource_type"),
        payload.get("locale"),
        len(translations),
    )
    return web.json_response({"translations": translations}, status=200)

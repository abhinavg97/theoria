from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from llm import (
    ProviderProcessError,
    _extract_claude_metadata,
    _extract_codex_metadata,
    _run_claude_streaming,
)
from observability import JsonFormatter, redact_text
from telemetry import aggregate_calls, enrich_call


class TelemetryTests(unittest.TestCase):
    def test_streamed_provider_failure_retains_events_and_usage(self):
        event = {
            "type": "result",
            "is_error": True,
            "result": "authentication failed",
            "total_cost_usd": 0,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
        raw_line = (json.dumps(event) + "\n").encode()

        class Stream:
            def __init__(self, lines=None, read_value=b""):
                self.lines = list(lines or [])
                self.read_value = read_value

            def __aiter__(self):
                self.iterator = iter(self.lines)
                return self

            async def __anext__(self):
                try:
                    return next(self.iterator)
                except StopIteration:
                    raise StopAsyncIteration

            async def read(self):
                return self.read_value

        class Process:
            returncode = 1
            stdout = Stream([raw_line])
            stderr = Stream(read_value=b"provider stderr")

            async def wait(self):
                return self.returncode

        with patch("llm._print_claude_event"):
            with self.assertRaises(ProviderProcessError) as caught:
                asyncio.run(_run_claude_streaming(Process(), schema=None))
        error = caught.exception
        self.assertEqual(error.raw_stdout, raw_line)
        self.assertEqual(error.raw_stderr, b"provider stderr")
        self.assertEqual(error.events, [event])

        call = {
            "backend": "claude",
            "model": "opus",
            "failed": True,
            **_extract_claude_metadata(event),
        }
        enrich_call(call)
        self.assertTrue(call["usage"]["complete"])
        self.assertEqual(call["usage"]["total_tokens"], 4)

    def test_codex_usage_preserves_cache_and_reasoning_dimensions(self):
        metadata = _extract_codex_metadata([
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 30,
                    "cached_input_tokens": 40,
                    "output_tokens_details": {"reasoning_tokens": 12},
                },
            },
        ])
        self.assertEqual(metadata["cache_read_input_tokens"], 40)
        self.assertEqual(metadata["reasoning_output_tokens"], 12)
        self.assertEqual(len(metadata["provider_usage"]), 1)

    def test_unpriced_call_is_explicitly_incomplete(self):
        call = enrich_call({
            "backend": "codex",
            "model": "gpt-test",
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 40,
        })
        self.assertEqual(call["usage"]["input_tokens"], 100)
        self.assertEqual(call["usage"]["output_tokens"], 20)
        self.assertIsNone(call["cost"]["amount_usd"])
        self.assertFalse(call["cost"]["complete"])

    def test_versioned_pricing_accounts_for_cache_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pricing.json"
            path.write_text(json.dumps({
                "pricing_id": "test-2026-01-01",
                "models": {
                    "codex:gpt-test": {
                        "input_token_semantics": "includes_cache_read",
                        "input_per_million_usd": 1,
                        "output_per_million_usd": 10,
                        "cache_read_per_million_usd": 0.1,
                        "cache_creation_per_million_usd": 2,
                    },
                },
            }))
            call = enrich_call({
                "backend": "codex",
                "model": "gpt-test",
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "cache_read_input_tokens": 1_000_000,
                "cache_creation_input_tokens": 1_000_000,
            }, pricing_path=str(path))
        self.assertEqual(call["cost"]["pricing_id"], "test-2026-01-01")
        self.assertEqual(call["cost"]["source"], "model_pricing_estimate")
        self.assertAlmostEqual(call["cost"]["amount_usd"], 12.1)

    def test_partial_cost_never_looks_like_complete_total(self):
        priced = enrich_call({
            "backend": "claude",
            "model": "opus",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_cost_usd": 0.25,
        })
        unknown = enrich_call({
            "backend": "codex",
            "model": "gpt-test",
            "input_tokens": 7,
            "output_tokens": 3,
            "failed": True,
        })
        metrics = aggregate_calls(
            [priced, unknown],
            problem_started_at="start",
            problem_ended_at="end",
            problem_duration_ms=100,
        )
        self.assertEqual(metrics["total_tokens"], 25)
        self.assertEqual(metrics["failed_calls"], 1)
        self.assertEqual(metrics["cost"]["priced_calls"], 1)
        self.assertEqual(metrics["cost"]["amount_usd"], 0.25)
        self.assertFalse(metrics["cost"]["complete"])
        self.assertIsNone(metrics["total_cost_usd"])


class LoggingTests(unittest.TestCase):
    def test_text_redaction_covers_common_inline_credentials(self):
        text = redact_text(
            "Authorization: Bearer abcdefghijklmnop api_key=supersecretvalue"
        )
        self.assertNotIn("abcdefghijklmnop", text)
        self.assertNotIn("supersecretvalue", text)
        self.assertEqual(text.count("[REDACTED]"), 2)

    def test_json_formatter_redacts_nested_secrets(self):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("test.theoria.redaction")
        logger.handlers[:] = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.info(
            "safe message",
            extra={
                "event": "test",
                "fields": {
                    "problem_id": "p1",
                    "nested": {"access_token": "do-not-log"},
                },
            },
        )
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["problem_id"], "p1")
        self.assertEqual(payload["nested"]["access_token"], "[REDACTED]")
        self.assertNotIn("do-not-log", stream.getvalue())


if __name__ == "__main__":
    unittest.main()

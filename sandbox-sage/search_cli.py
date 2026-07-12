#!/usr/bin/env python3
"""theoria-search — web search for sandboxed agents, as a plain CLI.

Local model providers (Ollama, LM Studio) reject Codex's provider-side
search tool types (`web_search`, MCP `namespace` tools) before the model
ever sees the prompt. A command on PATH needs no provider support at
all: any agent with a shell function tool can discover sources with

    theoria-search "original 1998 PageRank paper" --count 5

The query goes to the Brave Search API. BRAVE_API_KEY must be present in
the environment; the endpoint is fixed to Brave's hosted API. The
THEORIA_SEARCH_ENDPOINT override exists only so tests can point the CLI
at a mock server — anyone who can set it can already reach the network
directly, so it grants no new capability.

Stdlib-only on purpose: the sandbox images install it as a single file
at /usr/local/bin/theoria-search, and `pip install -e .` exposes the
same module as a console script for (explicitly opted-in) host runs.
This file is duplicated into sandbox/ and sandbox-sage/ because each
Docker build context is isolated; tests/test_search_cli.py fails if the
copies drift.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "1.0"
PROVIDER = "brave"
DEFAULT_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
API_KEY_ENV = "BRAVE_API_KEY"
ENDPOINT_ENV = "THEORIA_SEARCH_ENDPOINT"

REQUEST_TIMEOUT_SECS = 20
# The free Brave tier allows ~1 request/second; a burst from an agent
# loop surfaces as 429. Retry a couple of times before giving up.
RETRYABLE_STATUSES = {429, 503}
RETRY_MAX = 2
RETRY_DELAY_SECS = 1.1

# Exit codes, so agents and tests can react without parsing prose.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_HTTP = 4

_TAG = re.compile(r"<[^>]+>")
_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _clean(text) -> str:
    """Strip HTML tags/entities and control chars from API-provided text."""
    if not isinstance(text, str):
        return ""
    return _CTRL.sub("", html.unescape(_TAG.sub("", text))).strip()


def _endpoint() -> str:
    override = os.environ.get(ENDPOINT_ENV)
    if not override:
        return DEFAULT_ENDPOINT
    parsed = urllib.parse.urlsplit(override)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"{ENDPOINT_ENV} must be an absolute http(s) URL when set"
        )
    return override


def _request(url: str, api_key: str):
    return urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "X-Subscription-Token": api_key,
            "User-Agent": f"theoria-search/{VERSION}",
        },
        method="GET",
    )


def _fetch(query: str, count: int, offset: int, api_key: str) -> dict:
    params = urllib.parse.urlencode(
        {"q": query, "count": count, "offset": offset}
    )
    url = f"{_endpoint()}?{params}"
    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(
                _request(url, api_key), timeout=REQUEST_TIMEOUT_SECS,
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in RETRYABLE_STATUSES and attempts <= RETRY_MAX:
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), 30.0)
                except (TypeError, ValueError):
                    delay = RETRY_DELAY_SECS
                time.sleep(delay)
                continue
            body = ""
            try:
                body = _clean(error.read().decode(errors="replace"))[:300]
            except OSError:
                pass
            raise RuntimeError(
                f"search API returned HTTP {error.code}"
                + (f": {body}" if body else "")
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"search API is unreachable: {error}") from error
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"search API returned invalid JSON: {error}"
            ) from error


def _results(payload: dict) -> list[dict]:
    web = payload.get("web") if isinstance(payload, dict) else None
    raw = web.get("results") if isinstance(web, dict) else None
    results = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": _clean(item.get("title")),
            "url": _clean(item.get("url")),
            "description": _clean(item.get("description")),
            "age": _clean(item.get("age") or item.get("page_age")),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="theoria-search",
        description="Search the web (Brave Search API) and print result "
                    "titles, URLs, and snippets. Snippets are leads — fetch "
                    "and inspect the source before treating a claim as "
                    "verified.",
    )
    parser.add_argument("query", help="What to search for.")
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of results, 1-20 (default: 5).",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Result page offset, 0-9 (default: 0).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print raw JSON results instead of readable text.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"theoria-search {VERSION} ({PROVIDER})",
    )
    args = parser.parse_args(argv)

    if not args.query.strip():
        print("error: query must not be empty", file=sys.stderr)
        return EXIT_USAGE
    if not 1 <= args.count <= 20:
        print("error: --count must be between 1 and 20", file=sys.stderr)
        return EXIT_USAGE
    if not 0 <= args.offset <= 9:
        print("error: --offset must be between 0 and 9", file=sys.stderr)
        return EXIT_USAGE

    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(
            f"error: {API_KEY_ENV} is not set. Web search is unavailable in "
            "this environment — do not retry; verify the claim another way "
            "or take the conservative path.",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    try:
        payload = _fetch(args.query, args.count, args.offset, api_key)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_CONFIG
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_HTTP

    results = _results(payload)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return EXIT_OK
    if not results:
        print("No results.")
        return EXIT_OK
    print(f"# {len(results)} result(s) for: {_clean(args.query)}")
    for index, item in enumerate(results, start=1):
        age = f"  ({item['age']})" if item["age"] else ""
        print(f"{index}. {item['title']}{age}")
        print(f"   {item['url']}")
        if item["description"]:
            print(f"   {item['description']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

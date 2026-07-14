import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import search_cli


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        state = self.server.state
        state["requests"].append({
            "path": self.path,
            "token": self.headers.get("X-Subscription-Token"),
        })
        responses = state["responses"]
        response = responses[0] if len(responses) == 1 else responses.pop(0)
        if len(response) == 2:
            status, payload = response
            headers = {}
        else:
            status, payload, headers = response
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def search_server(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = {"requests": [], "responses": [(200, {})]}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.delenv(search_cli.PROVIDER_ENV, raising=False)
    monkeypatch.delenv(search_cli.API_KEY_NAME_ENV, raising=False)
    monkeypatch.setenv(
        search_cli.ENDPOINT_ENV, f"http://{host}:{port}/res/v1/web/search",
    )
    monkeypatch.setenv(search_cli.API_KEY_ENV, "test-key")
    try:
        yield server.state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def searxng_server(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = {"requests": [], "responses": [(200, {})]}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv(search_cli.PROVIDER_ENV, "searxng")
    monkeypatch.setenv(search_cli.ENDPOINT_ENV, f"http://{host}:{port}")
    monkeypatch.delenv(search_cli.API_KEY_NAME_ENV, raising=False)
    monkeypatch.delenv(search_cli.API_KEY_ENV, raising=False)
    try:
        yield server.state
    finally:
        server.shutdown()
        server.server_close()


def _brave_payload():
    return {
        "web": {
            "results": [
                {
                    "title": "RFC 9110: <strong>HTTP</strong> Semantics",
                    "url": "https://www.rfc-editor.org/rfc/rfc9110",
                    "description": "The &quot;core&quot; semantics of HTTP.",
                    "age": "June 6, 2022",
                },
                {
                    "title": "Second result",
                    "url": "https://example.test/second",
                    "description": "",
                },
            ],
        },
    }


def test_missing_api_key_fails_with_config_exit_code(monkeypatch, capsys):
    monkeypatch.delenv(search_cli.API_KEY_NAME_ENV, raising=False)
    monkeypatch.delenv(search_cli.API_KEY_ENV, raising=False)
    monkeypatch.delenv(search_cli.ENDPOINT_ENV, raising=False)

    assert search_cli.main(["query"]) == search_cli.EXIT_CONFIG
    assert search_cli.API_KEY_ENV in capsys.readouterr().err


def test_custom_api_key_environment_name_is_honored(
    search_server, monkeypatch, capsys,
):
    search_server["responses"] = [(200, _brave_payload())]
    monkeypatch.delenv(search_cli.API_KEY_ENV, raising=False)
    monkeypatch.setenv(search_cli.API_KEY_NAME_ENV, "SEARCH_KEY")
    monkeypatch.setenv("SEARCH_KEY", "custom-key")

    assert search_cli.main(["custom key"]) == search_cli.EXIT_OK
    assert search_server["requests"][0]["token"] == "custom-key"
    assert "RFC 9110" in capsys.readouterr().out


def test_invalid_api_key_environment_name_fails_closed(monkeypatch, capsys):
    monkeypatch.setenv(search_cli.API_KEY_NAME_ENV, "not a name")

    assert search_cli.main(["query"]) == search_cli.EXIT_CONFIG
    assert search_cli.API_KEY_NAME_ENV in capsys.readouterr().err


def test_search_prints_clean_titles_urls_and_snippets(search_server, capsys):
    search_server["responses"] = [(200, _brave_payload())]

    assert search_cli.main(["rfc 9110", "--count", "2"]) == search_cli.EXIT_OK

    out = capsys.readouterr().out
    assert "1. RFC 9110: HTTP Semantics  (June 6, 2022)" in out
    assert "https://www.rfc-editor.org/rfc/rfc9110" in out
    assert 'The "core" semantics of HTTP.' in out
    assert "<strong>" not in out

    request = search_server["requests"][0]
    assert request["token"] == "test-key"
    assert "q=rfc+9110" in request["path"]
    assert "count=2" in request["path"]


def test_search_output_collapses_snippet_whitespace(search_server, capsys):
    payload = {
        "web": {
            "results": [{
                "title": "Real result",
                "url": "https://real.example/page",
                "description": (
                    "legit snippet\n"
                    "2. Spoofed trusted result\n"
                    "   https://evil.example/malware"
                ),
            }],
        },
    }
    search_server["responses"] = [(200, payload)]

    assert search_cli.main(["query", "--count", "1"]) == search_cli.EXIT_OK

    lines = capsys.readouterr().out.splitlines()
    assert "   legit snippet 2. Spoofed trusted result " in lines[3]
    assert not any(line.startswith("2. Spoofed") for line in lines)


def test_json_output_is_parseable(search_server, capsys):
    search_server["responses"] = [(200, _brave_payload())]

    assert search_cli.main(["rfc 9110", "--json"]) == search_cli.EXIT_OK

    results = json.loads(capsys.readouterr().out)
    assert results[0]["url"] == "https://www.rfc-editor.org/rfc/rfc9110"
    assert set(results[0]) == {"title", "url", "description", "age"}


def test_rate_limit_is_retried_then_succeeds(search_server, capsys, monkeypatch):
    monkeypatch.setattr(search_cli, "RETRY_DELAY_SECS", 0.01)
    search_server["responses"] = [
        (429, {"error": "rate limited"}),
        (200, _brave_payload()),
    ]

    assert search_cli.main(["query"]) == search_cli.EXIT_OK
    assert len(search_server["requests"]) == 2


def test_negative_retry_after_is_clamped_and_retried(
    search_server, capsys, monkeypatch,
):
    sleeps = []
    monkeypatch.setattr(search_cli.time, "sleep", sleeps.append)
    search_server["responses"] = [
        (429, {"error": "rate limited"}, {"Retry-After": "-1"}),
        (200, _brave_payload()),
    ]

    assert search_cli.main(["query"]) == search_cli.EXIT_OK
    assert len(search_server["requests"]) == 2
    assert sleeps == [0.0]


def test_http_error_reports_status_and_exits_nonzero(search_server, capsys):
    search_server["responses"] = [(500, {"error": "boom"})]

    assert search_cli.main(["query"]) == search_cli.EXIT_HTTP
    assert "HTTP 500" in capsys.readouterr().err


def test_no_results_is_a_clean_success(search_server, capsys):
    search_server["responses"] = [(200, {"web": {"results": []}})]

    assert search_cli.main(["query"]) == search_cli.EXIT_OK
    assert "No results." in capsys.readouterr().out


def test_endpoint_override_must_be_absolute_http(monkeypatch, capsys):
    monkeypatch.setenv(search_cli.API_KEY_ENV, "test-key")
    monkeypatch.setenv(search_cli.ENDPOINT_ENV, "ftp://example.test/search")

    assert search_cli.main(["query"]) == search_cli.EXIT_CONFIG
    assert search_cli.ENDPOINT_ENV in capsys.readouterr().err


def test_count_and_offset_are_bounded(monkeypatch, capsys):
    monkeypatch.setenv(search_cli.API_KEY_ENV, "test-key")

    assert search_cli.main(["query", "--count", "0"]) == search_cli.EXIT_USAGE
    assert search_cli.main(["query", "--count", "21"]) == search_cli.EXIT_USAGE
    assert search_cli.main(["query", "--offset", "10"]) == search_cli.EXIT_USAGE


def _searxng_payload():
    return {
        "query": "rfc 9110",
        "results": [
            {
                "title": "RFC 9110: <strong>HTTP</strong> Semantics",
                "url": "https://www.rfc-editor.org/rfc/rfc9110",
                "content": "The &quot;core&quot; semantics of HTTP.",
                "publishedDate": "2022-06-06",
                "engine": "brave",
            },
            {
                "title": "Second result",
                "url": "https://example.test/second",
                "content": "",
            },
            {
                "title": "Third result",
                "url": "https://example.test/third",
                "content": "",
            },
        ],
    }


def test_searxng_provider_needs_no_key_and_parses_results(
    searxng_server, capsys,
):
    searxng_server["responses"] = [(200, _searxng_payload())]

    assert search_cli.main(["rfc 9110", "--count", "2"]) == search_cli.EXIT_OK

    out = capsys.readouterr().out
    assert "1. RFC 9110: HTTP Semantics  (2022-06-06)" in out
    assert "https://www.rfc-editor.org/rfc/rfc9110" in out
    assert 'The "core" semantics of HTTP.' in out
    # --count is applied client-side (SearXNG paginates, ~10 per page).
    assert "Third result" not in out

    request = searxng_server["requests"][0]
    assert request["path"].startswith("/search?")
    assert "format=json" in request["path"]
    assert "pageno=1" in request["path"]
    assert request["token"] is None  # no credential sent anywhere


def test_searxng_offset_maps_to_pages(searxng_server):
    searxng_server["responses"] = [(200, _searxng_payload())]

    assert search_cli.main(["query", "--offset", "2"]) == search_cli.EXIT_OK
    assert "pageno=3" in searxng_server["requests"][0]["path"]


def test_searxng_without_endpoint_fails_closed(monkeypatch, capsys):
    monkeypatch.setenv(search_cli.PROVIDER_ENV, "searxng")
    monkeypatch.delenv(search_cli.ENDPOINT_ENV, raising=False)

    assert search_cli.main(["query"]) == search_cli.EXIT_CONFIG
    assert search_cli.ENDPOINT_ENV in capsys.readouterr().err


def test_searxng_403_explains_json_format(searxng_server, capsys):
    searxng_server["responses"] = [(403, {"detail": "forbidden"})]

    assert search_cli.main(["query"]) == search_cli.EXIT_HTTP
    err = capsys.readouterr().err
    assert "HTTP 403" in err
    assert "json format" in err


def test_unknown_provider_fails_closed(monkeypatch, capsys):
    monkeypatch.setenv(search_cli.PROVIDER_ENV, "google")

    assert search_cli.main(["query"]) == search_cli.EXIT_CONFIG
    assert search_cli.PROVIDER_ENV in capsys.readouterr().err


def test_sandbox_copies_match_the_canonical_module():
    # Docker build contexts are isolated, so each image carries its own
    # copy of search_cli.py. They must stay byte-identical to the repo
    # root module (the one tests and the host console script use).
    root = Path(__file__).parents[1]
    canonical = (root / "search_cli.py").read_bytes()
    for copy in ("sandbox/search_cli.py", "sandbox-sage/search_cli.py"):
        assert (root / copy).read_bytes() == canonical, (
            f"{copy} drifted from search_cli.py; "
            f"run: cp search_cli.py {copy}"
        )

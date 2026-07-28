"""The Docker preflight extracts a plaintext Claude OAuth token from the
macOS Keychain into $TMPDIR. Whatever happens next, that file has to be
removed — including when a preflight check itself raises."""

import asyncio

import pytest

import harness


class FakeSandbox:
    """Stand-in for the `sandbox` module, recording what got cleaned up."""

    DEFAULT_IMAGE = "theoria-sandbox:latest"

    def __init__(self, digest="sha256:abc", config_path="/tmp/fake-config.json"):
        self._digest = digest
        self._config_path = config_path
        self.creds_extracted = 0
        self.cleaned_creds = []
        self.cleaned_configs = []

    def image_digest(self, image):
        return self._digest

    def refresh_claude_credentials(self):
        self.creds_extracted += 1
        return "/tmp/fake-creds.json"

    def prepare_claude_config(self):
        return self._config_path

    def cleanup_credentials(self, path):
        self.cleaned_creds.append(path)

    def cleanup_claude_config(self, path):
        self.cleaned_configs.append(path)

    def container_tool_versions(self, image):
        return {}


@pytest.fixture
def docker_run(tmp_path, monkeypatch):
    """A cwd-isolated run_problems in Docker mode, with a fake sandbox."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(harness._args_ref, "docker", True)
    monkeypatch.setitem(harness._args_ref, "image", "theoria-sandbox-sage:latest")

    def _run(fake):
        monkeypatch.setattr(harness, "sbx", fake)
        return asyncio.run(harness.run_problems([], 1, "runs/test_run.json"))

    return _run


def test_missing_image_never_touches_the_keychain(docker_run):
    fake = FakeSandbox(digest=None)

    with pytest.raises(RuntimeError, match="not available locally"):
        docker_run(fake)

    # The cheap check runs first, so no token is extracted at all.
    assert fake.creds_extracted == 0
    assert fake.cleaned_creds == []


def test_failed_config_snapshot_still_wipes_extracted_credentials(docker_run):
    # Image is fine and the token gets extracted; the *next* check fails.
    fake = FakeSandbox(digest="sha256:abc", config_path=None)

    with pytest.raises(RuntimeError, match="could not snapshot"):
        docker_run(fake)

    assert fake.creds_extracted == 1
    assert fake.cleaned_creds == ["/tmp/fake-creds.json"]


def test_credentials_are_wiped_on_the_success_path_too(docker_run):
    fake = FakeSandbox()

    assert docker_run(fake) == []

    assert fake.cleaned_creds == ["/tmp/fake-creds.json"]
    assert fake.cleaned_configs == ["/tmp/fake-config.json"]

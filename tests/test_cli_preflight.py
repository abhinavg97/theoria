import argparse

import pytest

import cli


def _args(docker=None, image=None):
    return argparse.Namespace(docker=docker, image=image)


def test_resolve_sandbox_defaults_to_docker_on_sage_image():
    assert cli._resolve_sandbox(_args()) == (True, cli.SAGE_IMAGE)


def test_resolve_sandbox_honors_explicit_flags():
    assert cli._resolve_sandbox(_args(docker=False)) == (False, cli.SAGE_IMAGE)
    assert cli._resolve_sandbox(_args(image="custom:tag")) == (True, "custom:tag")


def test_preflight_exits_when_image_missing(monkeypatch, capsys):
    monkeypatch.setattr(cli.sandbox, "image_digest", lambda image: None)

    with pytest.raises(SystemExit) as excinfo:
        cli._preflight(_args())

    # The message has to name the image and the command that builds it —
    # this error is the first thing a new user hits.
    message = str(excinfo.value)
    assert cli.SAGE_IMAGE in message
    assert "theoria build --sage" in message


def test_preflight_suggests_standard_build_for_standard_image(monkeypatch):
    monkeypatch.setattr(cli.sandbox, "image_digest", lambda image: None)

    with pytest.raises(SystemExit) as excinfo:
        cli._preflight(_args(image=cli.STD_IMAGE))

    assert "theoria build --standard" in str(excinfo.value)


def test_preflight_suggests_docker_build_for_unknown_image(monkeypatch):
    monkeypatch.setattr(cli.sandbox, "image_digest", lambda image: None)

    with pytest.raises(SystemExit) as excinfo:
        cli._preflight(_args(image="my-own:tag"))

    assert "docker build -t my-own:tag" in str(excinfo.value)


def test_preflight_passes_when_image_present(monkeypatch):
    monkeypatch.setattr(cli.sandbox, "image_digest", lambda image: "sha256:abc")

    cli._preflight(_args())  # no SystemExit


def test_preflight_skipped_entirely_without_docker(monkeypatch):
    def _fail(image):
        raise AssertionError("image_digest must not be consulted with --no-docker")

    monkeypatch.setattr(cli.sandbox, "image_digest", _fail)

    cli._preflight(_args(docker=False))

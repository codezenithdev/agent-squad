"""DockerSandbox: result handling (no Docker) + a Docker-gated real run."""
import subprocess
import types

import pytest

import core.sandbox as sb_mod
from core.sandbox import DockerSandbox, SandboxResult, docker_available


# --- pure result logic (no Docker) -----------------------------------------

def test_result_ok_and_tail():
    r = SandboxResult("cmd", 0, "all good", "", False)
    assert r.ok and "all good" in r.tail()
    big = SandboxResult("cmd", 1, "x" * 5000, "", False)
    assert not big.ok
    assert len(big.tail(100)) <= 103  # "..." + last 100 chars


def test_run_handles_missing_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sb_mod.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no docker")),
    )
    res = DockerSandbox("python:3.11-slim").run(str(tmp_path), "echo hi")
    assert res.exit_code == 127 and not res.ok


def test_run_handles_timeout(monkeypatch, tmp_path):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(sb_mod.subprocess, "run", slow)
    res = DockerSandbox("python:3.11-slim", timeout=1).run(str(tmp_path), "sleep 999")
    assert res.timed_out and res.exit_code == 124


def test_run_success(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sb_mod.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="PASSED", stderr=""),
    )
    res = DockerSandbox("python:3.11-slim").run(str(tmp_path), "pytest")
    assert res.ok and "PASSED" in res.stdout


# --- real Docker run (skipped unless the daemon is up) ---------------------

@pytest.mark.skipif(not docker_available(), reason="Docker daemon not available")
def test_real_pytest_detects_pass_and_fail(tmp_path):
    from config import get_settings

    image = get_settings().sandbox_python_image
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text("pytest\n")
    (backend / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")

    passed = DockerSandbox(image, timeout=180).run(
        str(tmp_path), "cd backend && pip install -q -r requirements.txt && python -m pytest -q"
    )
    assert passed.ok, passed.tail()

    (backend / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    failed = DockerSandbox(image, timeout=180).run(
        str(tmp_path), "cd backend && python -m pytest -q"
    )
    assert not failed.ok

"""Git integration (v2.5): committing a workspace to a branch."""
import subprocess

import pytest

from core.git_ops import commit_workspace, git_available


@pytest.mark.skipif(not git_available(), reason="git not available")
def test_commit_workspace(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "main.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# generated\n")

    res = commit_workspace(str(tmp_path), "agent/test-branch", "feat: initial")

    assert res["committed"] is True
    assert res["branch"] == "agent/test-branch"
    assert res["commit"]  # non-empty short sha

    # The repo really exists with that branch and the files are tracked.
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(tmp_path), capture_output=True, text=True,
    ).stdout.strip()
    assert branch == "agent/test-branch"
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(tmp_path), capture_output=True, text=True
    ).stdout
    assert "backend/main.py" in tracked


def test_commit_missing_workspace_is_graceful(tmp_path):
    res = commit_workspace(str(tmp_path / "does-not-exist"), "agent/x", "msg")
    assert res["committed"] is False and res["reason"]

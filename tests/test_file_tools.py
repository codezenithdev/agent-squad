"""Workspace file tools: write/read/list, the traversal guard, and the digest."""
from core.file_tools import (
    list_workspace_files,
    make_file_tools,
    read_workspace_digest,
)


def _tools(tmp_path):
    return {t.name: t for t in make_file_tools(str(tmp_path))}


def test_write_then_read_and_list(tmp_path):
    tools = _tools(tmp_path)
    tools["write_file"].invoke({"path": "backend/main.py", "content": "print('hi')"})
    assert "wrote backend/main.py" in tools["write_file"].invoke(
        {"path": "backend/main.py", "content": "print('hi')"}
    )
    assert tools["read_file"].invoke({"path": "backend/main.py"}) == "print('hi')"
    assert "backend/main.py" in list_workspace_files(str(tmp_path))


def test_traversal_is_blocked(tmp_path):
    tools = _tools(tmp_path)
    result = tools["write_file"].invoke({"path": "../escape.txt", "content": "x"})
    assert result.startswith("ERROR")


def test_read_missing_file(tmp_path):
    tools = _tools(tmp_path)
    assert tools["read_file"].invoke({"path": "nope.py"}).startswith("ERROR")


def test_digest_concatenates_with_headers(tmp_path):
    tools = _tools(tmp_path)
    tools["write_file"].invoke({"path": "a.py", "content": "AAA"})
    tools["write_file"].invoke({"path": "b.py", "content": "BBB"})
    digest = read_workspace_digest(str(tmp_path))
    assert "----- a.py -----" in digest and "AAA" in digest
    assert "----- b.py -----" in digest and "BBB" in digest


def test_digest_empty_workspace():
    assert read_workspace_digest("") == ""

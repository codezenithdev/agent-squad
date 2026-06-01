"""Web search tool wiring (v2.4): tool shape + provider-gated binding."""
import core.llm as llm_module
from config import Settings
from core.llm import complete, web_search_tool


class _FakeChat:
    """Records bind_tools calls and returns a canned response from ainvoke."""

    def __init__(self):
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content="ok")


def test_web_search_tool_shape(monkeypatch):
    s = Settings(_env_file=None, web_search_tool_type="web_search_20250305", web_search_max_uses=5)
    monkeypatch.setattr(llm_module, "get_settings", lambda: s)
    assert web_search_tool() == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
    }


async def test_search_binds_on_anthropic(monkeypatch):
    s = Settings(
        _env_file=None, use_mock_llm=False, llm_provider="anthropic",
        anthropic_api_key="k", enable_web_search=True,
    )
    monkeypatch.setattr(llm_module, "get_settings", lambda: s)
    fake = _FakeChat()
    monkeypatch.setattr(llm_module, "_get_chat_model", lambda *a, **k: fake)

    out = await complete("frontend", "sys", "user", web_search=True)
    assert out == "ok"
    assert fake.bound and fake.bound[0]["name"] == "web_search"


async def test_search_is_noop_on_openai(monkeypatch):
    s = Settings(
        _env_file=None, use_mock_llm=False, llm_provider="openai", openai_api_key="k"
    )
    monkeypatch.setattr(llm_module, "get_settings", lambda: s)
    fake = _FakeChat()
    monkeypatch.setattr(llm_module, "_get_chat_model", lambda *a, **k: fake)

    await complete("frontend", "sys", "user", web_search=True)
    assert fake.bound is None  # no web search tool bound on OpenAI


async def test_search_disabled_by_config(monkeypatch):
    s = Settings(
        _env_file=None, use_mock_llm=False, llm_provider="anthropic",
        anthropic_api_key="k", enable_web_search=False,
    )
    monkeypatch.setattr(llm_module, "get_settings", lambda: s)
    fake = _FakeChat()
    monkeypatch.setattr(llm_module, "_get_chat_model", lambda *a, **k: fake)

    await complete("frontend", "sys", "user", web_search=True)
    assert fake.bound is None  # disabled via enable_web_search=False

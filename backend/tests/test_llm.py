"""
Tests for app.llm — all OpenAI API calls are mocked so no live key is needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_response(content):
    """Fake OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_openai(content):
    """Patch openai.AsyncOpenAI so it returns a client whose create() yields content."""
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_make_response(content))
    return mock_client


# ── generate() routing ─────────────────────────────────────────────────────────

class TestGenerate:
    @pytest.mark.asyncio
    async def test_api_mode_routes_to_generate_api(self):
        with patch("app.llm.settings") as s, \
             patch("app.llm._generate_api", new_callable=AsyncMock) as mock_api:
            s.LLM_MODE = "api"
            mock_api.return_value = "api response"
            from app import llm
            result = await llm.generate("hello")
            mock_api.assert_called_once_with("hello", 300, 0.0, None)
            assert result == "api response"

    @pytest.mark.asyncio
    async def test_local_mode_routes_to_generate_local(self):
        with patch("app.llm.settings") as s, \
             patch("app.llm._generate_local", new_callable=AsyncMock) as mock_local:
            s.LLM_MODE = "local"
            mock_local.return_value = "local response"
            from app import llm
            result = await llm.generate("hello", max_tokens=100)
            mock_local.assert_called_once_with("hello", 100, 0.0, None)
            assert result == "local response"

    @pytest.mark.asyncio
    async def test_passes_all_params(self):
        with patch("app.llm.settings") as s, \
             patch("app.llm._generate_api", new_callable=AsyncMock) as mock_api:
            s.LLM_MODE = "api"
            mock_api.return_value = ""
            from app import llm
            await llm.generate("p", max_tokens=50, temperature=0.5, system_message="sys")
            mock_api.assert_called_once_with("p", 50, 0.5, "sys")


# ── _generate_api() ────────────────────────────────────────────────────────────

class TestGenerateApi:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        mock_client = _mock_openai("Result text")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            result = await llm._generate_api("prompt", 300, 0.0, None)
            assert result == "Result text"

    @pytest.mark.asyncio
    async def test_includes_user_message(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            await llm._generate_api("my prompt", 300, 0.0, None)
            call_msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
            assert call_msgs[-1] == {"role": "user", "content": "my prompt"}

    @pytest.mark.asyncio
    async def test_prepends_system_message(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            await llm._generate_api("prompt", 300, 0.0, "be helpful")
            msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
            assert msgs[0] == {"role": "system", "content": "be helpful"}
            assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_no_system_message_sends_only_user(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            await llm._generate_api("prompt", 300, 0.0, None)
            msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
            assert len(msgs) == 1
            assert msgs[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        with patch("app.llm.settings") as s:
            s.OPENAI_API_KEY = None
            from app import llm
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                await llm._generate_api("prompt", 300, 0.0, None)

    @pytest.mark.asyncio
    async def test_none_content_returns_empty_string(self):
        mock_client = _mock_openai(None)
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            result = await llm._generate_api("prompt", 300, 0.0, None)
            assert result == ""

    @pytest.mark.asyncio
    async def test_passes_model_and_params(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o-mini"
            from app import llm
            await llm._generate_api("p", 150, 0.3, None)
            kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == "gpt-4o-mini"
            assert kwargs["max_tokens"] == 150
            assert kwargs["temperature"] == 0.3


# ── _generate_local() ──────────────────────────────────────────────────────────

class TestGenerateLocal:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        mock_client = _mock_openai("Local result")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.LOCAL_MODEL_URL = "http://localhost:8080/v1"
            s.LOCAL_MODEL_NAME = "gpt-oss-20b"
            from app import llm
            result = await llm._generate_local("test", 200, 0.0, None)
            assert result == "Local result"

    @pytest.mark.asyncio
    async def test_uses_local_url_and_model(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client) as MockOpenAI:
            s.LOCAL_MODEL_URL = "http://localhost:8080/v1"
            s.LOCAL_MODEL_NAME = "my-local-model"
            from app import llm
            await llm._generate_local("prompt", 200, 0.0, None)
            # AsyncOpenAI should be created with base_url (not api_key from settings)
            init_kwargs = MockOpenAI.call_args.kwargs
            assert init_kwargs.get("base_url") == "http://localhost:8080/v1"
            # Model in completions.create should be local model name
            kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == "my-local-model"

    @pytest.mark.asyncio
    async def test_with_system_message(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.LOCAL_MODEL_URL = "http://localhost:8080/v1"
            s.LOCAL_MODEL_NAME = "local"
            from app import llm
            await llm._generate_local("prompt", 200, 0.0, "system msg")
            msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == "system msg"

    @pytest.mark.asyncio
    async def test_none_content_returns_empty_string(self):
        mock_client = _mock_openai(None)
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.LOCAL_MODEL_URL = "http://localhost:8080/v1"
            s.LOCAL_MODEL_NAME = "local"
            from app import llm
            result = await llm._generate_local("p", 200, 0.0, None)
            assert result == ""


# ── generate_api_only() ────────────────────────────────────────────────────────

class TestGenerateApiOnly:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        mock_client = _mock_openai("Triage JSON")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            result = await llm.generate_api_only("triage prompt")
            assert result == "Triage JSON"

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        with patch("app.llm.settings") as s:
            s.OPENAI_API_KEY = None
            from app import llm
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                await llm.generate_api_only("prompt")

    @pytest.mark.asyncio
    async def test_with_system_message(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            await llm.generate_api_only("p", system_message="sys")
            msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_default_max_tokens(self):
        mock_client = _mock_openai("ok")
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            await llm.generate_api_only("p")
            kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert kwargs["max_tokens"] == 400

    @pytest.mark.asyncio
    async def test_none_content_returns_empty_string(self):
        mock_client = _mock_openai(None)
        with patch("app.llm.settings") as s, \
             patch("openai.AsyncOpenAI", return_value=mock_client):
            s.OPENAI_API_KEY = "sk-test"
            s.OPENAI_MODEL = "gpt-4o"
            from app import llm
            result = await llm.generate_api_only("p")
            assert result == ""

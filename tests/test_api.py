import importlib

import pytest
from fastapi.testclient import TestClient


class FakeChain:
    def __init__(self, reply="fake answer"):
        self.reply = reply
        self.calls = []
        self.refresh_calls = 0

    async def astream(self, question, chat_history=None):
        self.calls.append((question, chat_history))
        for token in self.reply.split(" "):
            yield token + " "

    def refresh_retriever(self):
        self.refresh_calls += 1


@pytest.fixture
def api_module(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    import api as api_module

    importlib.reload(api_module)
    yield api_module


@pytest.fixture
def client(api_module):
    fake_chain = FakeChain()
    api_module.app.dependency_overrides[api_module.get_chain] = lambda: fake_chain
    with TestClient(api_module.app) as c:
        c.fake_chain = fake_chain
        yield c
    api_module.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_streams_answer_and_forwards_history(client):
    history = [{"role": "user", "content": "earlier question"}]
    r = client.post("/ask", json={"question": "What is Act 1040?", "chat_history": history})

    assert r.status_code == 200
    assert "fake answer" in r.text
    assert client.fake_chain.calls[0][0] == "What is Act 1040?"
    assert client.fake_chain.calls[0][1] == history


def test_ask_without_api_key_configured_allows_request(client):
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200


def test_ask_rejects_missing_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    import api as api_module

    importlib.reload(api_module)
    api_module.app.dependency_overrides[api_module.get_chain] = lambda: FakeChain()

    with TestClient(api_module.app) as c:
        r = c.post("/ask", json={"question": "hi"})
        assert r.status_code == 401

        r_ok = c.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "secret123"})
        assert r_ok.status_code == 200

    api_module.app.dependency_overrides.clear()
    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(api_module)


def test_ask_rejects_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_rejects_oversized_question(client):
    r = client.post("/ask", json={"question": "x" * 2001})
    assert r.status_code == 422


def test_ask_rejects_invalid_history_role(client):
    r = client.post("/ask", json={"question": "hi", "chat_history": [{"role": "system", "content": "x"}]})
    assert r.status_code == 422


def test_ask_rejects_oversized_history(client):
    history = [{"role": "user", "content": "x"} for _ in range(41)]
    r = client.post("/ask", json={"question": "hi", "chat_history": history})
    assert r.status_code == 422


def test_admin_refresh_calls_chain_refresh(client):
    r = client.post("/admin/refresh")
    assert r.status_code == 200
    assert r.json() == {"status": "refreshed"}
    assert client.fake_chain.refresh_calls == 1


def test_admin_refresh_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    import api as api_module

    importlib.reload(api_module)
    fake_chain = FakeChain()
    api_module.app.dependency_overrides[api_module.get_chain] = lambda: fake_chain

    with TestClient(api_module.app) as c:
        assert c.post("/admin/refresh").status_code == 401
        r_ok = c.post("/admin/refresh", headers={"X-API-Key": "secret123"})
        assert r_ok.status_code == 200
        assert fake_chain.refresh_calls == 1

    api_module.app.dependency_overrides.clear()
    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(api_module)


def test_health_ready_when_ollama_reachable(client, monkeypatch, api_module):
    class FakeResponse:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(api_module.requests, "get", lambda *a, **k: FakeResponse())

    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_health_ready_when_ollama_unreachable(client, monkeypatch, api_module):
    def raise_connection_error(*a, **k):
        raise api_module.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(api_module.requests, "get", raise_connection_error)

    r = client.get("/health/ready")
    assert r.status_code == 503


def test_ask_rate_limit_returns_429(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("ASK_RATE_LIMIT", "2/minute")
    import api as api_module

    importlib.reload(api_module)
    api_module.app.dependency_overrides[api_module.get_chain] = lambda: FakeChain()

    with TestClient(api_module.app) as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 200
        assert c.post("/ask", json={"question": "hi"}).status_code == 200
        r = c.post("/ask", json={"question": "hi"})
        assert r.status_code == 429

    api_module.app.dependency_overrides.clear()
    monkeypatch.delenv("ASK_RATE_LIMIT", raising=False)
    importlib.reload(api_module)

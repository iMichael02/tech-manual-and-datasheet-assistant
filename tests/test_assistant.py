import pytest

from assistant.ingest import chunk_page
from assistant.pipeline import Assistant


@pytest.fixture(scope="module")
def a():
    return Assistant.build("extractive")


def test_chunking_overlaps_and_skips_tiny_pages():
    assert chunk_page("too short") == []
    chunks = chunk_page(" ".join(f"w{i}" for i in range(400)), size=100, overlap=20)
    assert chunks[1].split()[0] == "w80"


def test_part_routing_restricts_to_right_document(a):
    r = a.retriever.search("What is the I2C address of the BMP280?")
    assert r.parts == ["BMP280"]
    assert {h.chunk.doc for h in r.hits} == {"BST-BMP280-DS001-11.pdf"}


def test_alias_resolves_misnamed_file(a):
    r = a.retriever.search("What is the I2C slave address of the MPL3115A2?")
    assert {h.chunk.doc for h in r.hits} == {"1893_datasheet.pdf"}


def test_unknown_part_is_refused(a):
    ans = a.ask("What is the I2C address of the BME680?")
    assert ans.refused and "BME680" in ans.text


@pytest.mark.parametrize("q", ["What is the capital of France?", "Give me a recipe for pancakes."])
def test_out_of_domain_is_refused(a, q):
    assert a.ask(q).refused


def test_answer_carries_page_citations(a):
    ans = a.ask("What is the maximum conversion time of the DS18B20 at 12-bit resolution?")
    assert not ans.refused and ans.sources and "p." in ans.sources[0]


class _Spy:
    name = "spy"

    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def answer(self, question, r, history=None):
        self.seen.append(r)
        return self.reply


def test_norag_skips_retrieval_and_scope_gate(a):
    from assistant.generate import REFUSAL_PREFIX
    spy = _Spy("The MAX31855 resolution is 0.25 C.")
    b = Assistant(a.retriever, spy)
    ans = b.ask("What is the capital of France?", rag=False)   # would be refused by the scope gate with RAG
    assert spy.seen == [None] and ans.sources == [] and not ans.refused
    spy.reply = REFUSAL_PREFIX + " (not covered)."
    assert b.ask("anything", rag=False).refused


def test_norag_prompt_has_no_excerpts_but_lists_documents():
    from assistant.generate import _user_msg, norag_system_prompt
    assert _user_msg("Q?", None) == "Q?"
    assert "DS18B20" in norag_system_prompt() and "Excerpts" not in norag_system_prompt()


# ---- OpenAPI tool server (needs fastapi + httpx: pip install -r requirements-server.txt) ----
def _client(monkeypatch, key=None):
    pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import tool_server
    if key: monkeypatch.setenv("TOOL_SERVER_API_KEY", key)
    else: monkeypatch.delenv("TOOL_SERVER_API_KEY", raising=False)
    return TestClient(tool_server.app)


def test_tool_server_returns_cited_excerpts(monkeypatch):
    c = _client(monkeypatch)
    j = c.post("/search_datasheets", json={"question": "What is the I2C address of the BMP280?"}).json()
    assert j["in_scope"] and j["detected_parts"] == ["BMP280"]
    assert [e["n"] for e in j["excerpts"]] == list(range(1, len(j["excerpts"]) + 1))
    assert all(e["file"] == "BST-BMP280-DS001-11.pdf" and e["page"] >= 1 for e in j["excerpts"])


def test_tool_server_refuses_out_of_scope_and_validates(monkeypatch):
    c = _client(monkeypatch)
    j = c.post("/search_datasheets", json={"question": "What is the capital of France?"}).json()
    assert j["in_scope"] is False and j["excerpts"] == [] and j["reason"]
    assert c.post("/search_datasheets", json={"question": "hi", "top_k": 99}).status_code == 422


def test_tool_server_api_key(monkeypatch):
    c = _client(monkeypatch, key="s3cret")
    body = {"question": "What is the I2C address of the BMP280?"}
    assert c.post("/search_datasheets", json=body).status_code == 401
    assert c.post("/search_datasheets", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/search_datasheets", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_tool_server_openapi_names_the_tool(monkeypatch):
    spec = _client(monkeypatch).get("/openapi.json").json()
    assert spec["paths"]["/search_datasheets"]["post"]["operationId"] == "search_datasheets"

"""OpenAPI tool server exposing the datasheet retriever to Open WebUI (or any OpenAPI-tool client).

Run:   uvicorn tool_server:app --app-dir apps --port 8000
Open WebUI: Settings -> Integrations -> add a Tool Server with URL http://localhost:8000
            (if Open WebUI runs in Docker use http://host.docker.internal:8000).
Auth:  set TOOL_SERVER_API_KEY to require "Authorization: Bearer <key>" (set the same key in Open WebUI).

Only retrieval + the scope gate live here; the chat model writes the answer from the returned excerpts.
"""
from __future__ import annotations

import hmac
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from assistant import ingest as ing
from assistant.pipeline import DATA_DIR, INDEX
from assistant.retrieval import Retriever

app = FastAPI(
    title="Datasheet Assistant Tools",
    version="1.0.0",
    description="Search the sensor/instrument datasheets and manuals in the knowledge base.",
)
_bearer = HTTPBearer(auto_error=False)


def check_auth(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    key = os.environ.get("TOOL_SERVER_API_KEY")
    if key and not (cred and hmac.compare_digest(cred.credentials, key)):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@lru_cache(maxsize=1)
def retriever() -> Retriever:
    if not INDEX.exists():
        ing.save(ing.ingest(DATA_DIR), INDEX)
    return Retriever(ing.load(INDEX))


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          description="The technical question, in natural language. Include the part number "
                                      "(e.g. BMP280, DS18B20, MAX31855) when the user mentions one.")
    top_k: int = Field(5, ge=1, le=10, description="Number of excerpts to return.")


class Excerpt(BaseModel):
    n: int = Field(..., description="Citation number to use in the answer as [n].")
    document: str = Field(..., description="Document title.")
    file: str
    page: int
    text: str


class SearchResponse(BaseModel):
    in_scope: bool = Field(..., description="False means the knowledge base does not cover this question: "
                                            "tell the user you cannot answer it from the documentation.")
    reason: str = Field("", description="Why the question was judged out of scope.")
    detected_parts: list[str]
    excerpts: list[Excerpt]


@app.post(
    "/search_datasheets",
    operation_id="search_datasheets",
    summary="Search the technical datasheets and manuals",
    description=(
        "Look up specifications, addresses, ranges, timings and troubleshooting steps for the electronic parts, "
        "sensors and instruments in the knowledge base (DS18B20, DHT22/AM2302/AM2303, HTU21D, HDC1008, BMP180/BMP280, "
        "MPL3115A2, MAX31855, TMP006, SLHT5, Rigol DS1000E). Returns numbered excerpts with document and page. "
        "ALWAYS call this before answering a technical question about these parts. Answer ONLY from the excerpts, "
        "cite them as [n] with document and page, quote numbers and units exactly, and if sources disagree say which "
        "document each value comes from. If in_scope is false or the excerpts lack the answer, say so instead of guessing."
    ),
    response_model=SearchResponse,
    dependencies=[Depends(check_auth)],
)
def search_datasheets(req: SearchRequest) -> SearchResponse:
    r = retriever().search(req.question, k=req.top_k)
    excerpts = [Excerpt(n=i, document=h.chunk.title, file=h.chunk.doc, page=h.chunk.page, text=h.chunk.text)
                for i, h in enumerate(r.hits, 1)] if r.in_scope else []
    return SearchResponse(in_scope=r.in_scope, reason=r.reason, detected_parts=r.parts, excerpts=excerpts)


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "chunks": len(retriever().chunks)}

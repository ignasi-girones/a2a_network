"""Integration tests for the file-attachment upload pipeline.

Covers the Fase 5 verification checklist:
  5. 6 archivos → backend rechaza con 400 indicando limite.
  6. Archivo .exe → rechazado por mime.
  7. Attachments persisted in DB → GET /debates/<id>/attachments returns them.
  Plus: multipart happy path, size limit, JSON fallback still works.
"""

from __future__ import annotations

import io
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agents.orchestrator.debate_routes import debate_routes
from agents.orchestrator.debate_store import DebateStore


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = DebateStore(db_path)
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
def app(store, tmp_path, monkeypatch):
    """Starlette app with debate routes, a temp store, and temp upload dir.

    ``_run_debate`` is patched to a no-op so the POST handler returns
    immediately without trying to start the real orchestrator pipeline.
    """
    monkeypatch.setattr(
        "agents.orchestrator.debate_routes.settings.attachments_storage_dir",
        str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(
        "agents.orchestrator.debate_routes._run_debate",
        AsyncMock(),
    )
    app = Starlette(routes=debate_routes)
    app.state.debate_store = store
    return app


def _make_file(name: str, content: bytes, mime: str):
    """Build a tuple suitable for httpx/TestClient multipart upload."""
    return ("files", (name, io.BytesIO(content), mime))


# ── JSON fallback (backwards compatibility) ────────────────────────────


def test_json_post_still_works(app):
    """The existing JSON path must keep working when no files are attached."""
    with TestClient(app) as client:
        r = client.post("/debates", json={"prompt": "test sin adjuntos"})
        assert r.status_code == 201
        body = r.json()
        assert "debate_id" in body
        assert body["attachments"] == 0


# ── Multipart happy path ───────────────────────────────────────────────


def test_multipart_single_txt_file(app):
    """A single .txt file is accepted and persisted."""
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "analiza este archivo"},
            files=[_make_file("notes.txt", b"hello world", "text/plain")],
        )
        assert r.status_code == 201
        body = r.json()
        assert body["attachments"] == 1

        # Verify GET /debates/<id>/attachments returns the file
        debate_id = body["debate_id"]
        r2 = client.get(f"/debates/{debate_id}/attachments")
        assert r2.status_code == 200
        att = r2.json()
        assert att["count"] == 1
        assert att["attachments"][0]["filename"] == "notes.txt"
        assert att["attachments"][0]["mime_type"] == "text/plain"
        assert att["attachments"][0]["size_bytes"] == 11


def test_multipart_csv_file(app):
    """CSV files are accepted."""
    csv_content = b"name,age\nAlice,30\nBob,25"
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "analiza estos datos"},
            files=[_make_file("data.csv", csv_content, "text/csv")],
        )
        assert r.status_code == 201
        assert r.json()["attachments"] == 1


def test_multipart_multiple_files(app):
    """Multiple files (up to the limit) are accepted."""
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "compara estos documentos"},
            files=[
                _make_file("a.txt", b"file a", "text/plain"),
                _make_file("b.txt", b"file b", "text/plain"),
                _make_file("c.csv", b"x,y\n1,2", "text/csv"),
            ],
        )
        assert r.status_code == 201
        assert r.json()["attachments"] == 3


# ── Checklist item 5: too many files → 400 ─────────────────────────────


def test_rejects_too_many_files(app):
    """Uploading more than attachments_max_files (5) must return 400."""
    files = [
        _make_file(f"file{i}.txt", b"data", "text/plain")
        for i in range(6)
    ]
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "demasiados archivos"},
            files=files,
        )
        assert r.status_code == 400
        body = r.json()
        assert body["error"] == "too_many_files"
        assert "5" in body["message"]


# ── Checklist item 6: .exe → rejected by mime ──────────────────────────


def test_rejects_exe_by_mime(app):
    """An .exe file (application/octet-stream) must be rejected."""
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "intento subir exe"},
            files=[_make_file("malware.exe", b"\x00\x01", "application/octet-stream")],
        )
        assert r.status_code == 400
        body = r.json()
        assert body["error"] == "invalid_mime"
        assert "malware.exe" in body["message"]


def test_rejects_docx_by_mime(app):
    """A .docx file is not in the allowed list."""
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "intento subir docx"},
            files=[_make_file(
                "doc.docx", b"fake",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )],
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_mime"


# ── Size limit ─────────────────────────────────────────────────────────


def test_rejects_oversized_file(app, monkeypatch):
    """A file exceeding attachments_max_size_mb must be rejected."""
    # Set a tiny limit for testing
    monkeypatch.setattr(
        "agents.orchestrator.debate_routes.settings.attachments_max_size_mb", 0
    )
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "archivo enorme"},
            files=[_make_file("big.txt", b"x" * 1024, "text/plain")],
        )
        assert r.status_code == 400
        assert r.json()["error"] == "file_too_large"


# ── Missing prompt in multipart ────────────────────────────────────────


def test_multipart_rejects_missing_prompt(app):
    """Multipart without a prompt field must return 400."""
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": ""},
            files=[_make_file("a.txt", b"data", "text/plain")],
        )
        assert r.status_code == 400
        assert r.json()["error"] == "missing_prompt"


# ── Checklist item 7: attachments survive in DB ────────────────────────


async def test_attachments_persist_in_db(store, app):
    """After a multipart POST, attachments are queryable from the store."""
    with TestClient(app) as client:
        r = client.post(
            "/debates",
            data={"prompt": "persistencia"},
            files=[
                _make_file("doc.txt", b"persisted content", "text/plain"),
                _make_file("data.csv", b"a,b\n1,2", "text/csv"),
            ],
        )
        assert r.status_code == 201
        debate_id = r.json()["debate_id"]

    # Query the store directly
    attachments = await store.get_attachments(debate_id)
    assert len(attachments) == 2
    filenames = {a.filename for a in attachments}
    assert filenames == {"doc.txt", "data.csv"}
    # Verify extracted_text was populated for plaintext
    txt_att = next(a for a in attachments if a.filename == "doc.txt")
    assert txt_att.extracted_text == "persisted content"


# ── GET /debates/<id>/attachments ──────────────────────────────────────


def test_attachments_endpoint_404_for_unknown(app):
    with TestClient(app) as client:
        r = client.get("/debates/nonexistent/attachments")
        assert r.status_code == 404


async def test_attachments_endpoint_empty(store, app):
    """A debate without attachments returns an empty list."""
    debate_id = await store.create_debate("sin adjuntos")
    with TestClient(app) as client:
        r = client.get(f"/debates/{debate_id}/attachments")
        assert r.status_code == 200
        assert r.json() == {"count": 0, "attachments": []}

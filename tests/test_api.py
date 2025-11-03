from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app


client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_context_tree():
    resp = client.post("/context/tree", json={"path": ".", "max_depth": 0, "max_entries": 5})
    data = resp.json()
    assert "entries" in data
    assert any(entry["type"] == "file" or entry["type"] == "dir" for entry in data["entries"])


def test_context_snippet():
    resp = client.post("/context/snippet", json={"path": "README.md", "line": 1, "context": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"].endswith("README.md")
    assert data["snippet"]


def test_fs_search():
    resp = client.post("/fs/search", json={"pattern": "Turkai", "path": "README.md", "max_results": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data


def test_context_diff():
    resp = client.post("/context/diff", json={"paths": [], "unified": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert "diff" in data

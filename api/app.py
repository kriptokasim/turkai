from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import json
import uuid
import logging
import requests

from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger("turkai")
logging.basicConfig(level=logging.INFO)

# Environment-variable names align with docker-compose and .env defaults.
MODEL = os.getenv("MODEL") or os.getenv("MODEL_NAME") or "qwen2.5-coder:1.5b"
OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "http://ollama:11434"
SESSION_DIR = Path(os.getenv("SESSION_DIR", "./data/sessions"))
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

SESSION_DIR.mkdir(parents=True, exist_ok=True)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatIn(BaseModel):
    messages: List[Message]
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    reset: Optional[bool] = False


app = FastAPI(title="Turkai API", version="0.2")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return {"name": "Turkai API", "model": MODEL, "tools_enabled": False, "tools": []}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/ingest")
def ingest_stub():
    return {"ok": False, "detail": "Not implemented yet (planned: RAG/ingest)"}


@app.get("/ui", include_in_schema=False)
def ui_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not bundled</h1>", status_code=404)


def _ollama_chat(payload: Dict[str, Any]) -> requests.Response:
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    return requests.post(url, json=payload, timeout=60)


def _ollama_generate(prompt: str, options: Dict[str, Any]) -> requests.Response:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    body = {"model": MODEL, "prompt": prompt, "stream": False, "options": options or {}}
    return requests.post(url, json=body, timeout=60)


def _extract_answer(resp_json: Dict[str, Any]) -> str:
    # Ollama /api/chat
    message = resp_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    # Some models return `messages`.
    messages = resp_json.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                text = msg.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    # Ollama /api/generate
    for key in ("response", "content", "text"):
        value = resp_json.get(key)
        if isinstance(value, str):
            return value

    return ""


def _session_path(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> Dict[str, Any]:
    try:
        with _session_path(session_id).open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {"messages": []}
    except json.JSONDecodeError:
        log.warning("Session %s is corrupt. Resetting.", session_id)
        return {"messages": []}


def _save_session(session_id: str, data: Dict[str, Any]) -> None:
    path = _session_path(session_id)
    tmp_path = path.with_suffix(".tmp")
    data["session_id"] = session_id
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp_path.replace(path)


def _summarize_session(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    last_user = next(
        (m.get("content") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    last_answer = next(
        (m.get("content") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"),
        None,
    )
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        updated_at = datetime.utcfromtimestamp(_session_path(session_id).stat().st_mtime).isoformat() + "Z"
    return {
        "session_id": session_id,
        "turns": len([m for m in messages if isinstance(m, dict) and m.get("role") == "user"]),
        "last_user": last_user,
        "last_answer": last_answer,
        "updated_at": updated_at,
    }


@app.post("/chat")
def chat(payload: ChatIn = Body(...)):
    """
    Return stable {"answer": "...", "session_id": "..."} responses so CLI stays predictable.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    history: List[Dict[str, str]] = []
    try:
        if payload.session_id and not payload.reset:
            stored = _load_session(session_id).get("messages", [])
            if isinstance(stored, list):
                history = [m for m in stored if isinstance(m, dict)]

        if payload.reset:
            history = []

        for message in payload.messages:
            history.append({"role": message.role, "content": message.content})

        options = payload.options or {}
        request_body = {
            "model": MODEL,
            "messages": history,
            "stream": bool(payload.stream),
            "options": options,
        }

        response = _ollama_chat(request_body)

        # Older Ollama builds may not expose /api/chat.
        if response.status_code == 404:
            last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
            response = _ollama_generate(last_user, options)

        if not response.ok:
            try:
                body = response.json()
            except Exception:  # noqa: BLE001
                body = response.text[:1000]
            log.error("Ollama upstream error %s: %s", response.status_code, body)
            return {
                "answer": f"[upstream {response.status_code}] {body}",
                "session_id": session_id,
                "messages": history,
            }

        try:
            data = response.json()
        except Exception:  # noqa: BLE001
            text = response.text if response.text else "[upstream returned non-JSON]"
            return {"answer": text[:1000], "session_id": session_id, "messages": history}

        answer = (_extract_answer(data) or "").strip()
        if not answer:
            return {"answer": "[empty answer]", "session_id": session_id, "messages": history}

        history.append({"role": "assistant", "content": answer})
        _save_session(session_id, {"messages": history})

        return {"answer": answer, "session_id": session_id, "messages": history}

    except requests.Timeout:
        log.exception("Upstream timeout")
        return {
            "answer": "[timeout] Ollama 60s içinde yanıt vermedi.",
            "session_id": session_id,
            "messages": history,
        }
    except requests.RequestException as exc:
        log.exception("Upstream request failed")
        return {"answer": f"[request error] {exc}", "session_id": session_id, "messages": history}
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled server error")
        return {"answer": f"[server error] {exc}", "session_id": session_id, "messages": history}


@app.get("/resume")
def resume(session_id: Optional[str] = Query(None), limit: int = Query(5, ge=1, le=50)):
    """
    Return summaries of existing sessions. If session_id is provided, return that session + its messages.
    """
    if session_id:
        data = _load_session(session_id)
        if not data.get("messages"):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        return {"session": _summarize_session(session_id, data), "messages": data.get("messages", [])}

    sessions = []
    for path in sorted(SESSION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        sid = path.stem
        try:
            payload = _load_session(sid)
            sessions.append(_summarize_session(sid, payload))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not summarize session %s: %s", sid, exc)
    return {"sessions": sessions}

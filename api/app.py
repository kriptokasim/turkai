# api/app.py
from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import os
import io
import re
import json
import uuid
import logging
import shlex
import subprocess

import requests
from fastapi import FastAPI, Body, HTTPException, Query, Header, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log = logging.getLogger("turkai")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Env / Paths
# -----------------------------------------------------------------------------
MODEL = os.getenv("MODEL") or os.getenv("MODEL_NAME") or "qwen2.5-coder:1.5b"
OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "http://ollama:11434"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

SESSION_DIR = Path(os.getenv("SESSION_DIR", "./data/sessions")).resolve()
SESSION_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(os.getenv("REPO_ROOT", ".")).resolve()

API_KEY = os.getenv("API_KEY", "").strip()  # optional; empty => auth disabled

# -----------------------------------------------------------------------------
# FastAPI app + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Turkai API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Auth helper (optional)
# -----------------------------------------------------------------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> bool:
    """
    If API_KEY is set, require X-API-Key for sensitive endpoints.
/fs, /exec, /agent are protected; /, /healthz, /chat, /chat/stream, /ui remain open.
    """
    if not API_KEY:
        return True
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    return True

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatIn(BaseModel):
    messages: List[Message]
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    reset: Optional[bool] = False

class FSWriteIn(BaseModel):
    path: str
    content: str

class ExecIn(BaseModel):
    cmd: str
    cwd: Optional[str] = None
    timeout: Optional[int] = None

class PytestIn(BaseModel):
    pattern: Optional[str] = None
    cwd: Optional[str] = None
    extra: Optional[str] = None  # e.g., "-q -k quick"

class AgentStepIn(BaseModel):
    session_id: Optional[str] = None
    action: Literal["fs.read", "fs.write", "exec.run", "finish"]
    args: Dict[str, Any] = {}

class AgentLoopIn(BaseModel):
    goal: str
    session_id: Optional[str] = None
    max_steps: int = 5

class AgentLoopStep(BaseModel):
    step: int
    action: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    observation: Optional[Any] = None
    raw: Optional[str] = None
    error: Optional[str] = None

class AgentLoopOut(BaseModel):
    session_id: str
    steps: List[AgentLoopStep]
    done: bool
    result: Optional[Any] = None

# -----------------------------------------------------------------------------
# Root / Health / UI
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "Turkai API", "model": MODEL, "tools_enabled": True, "tools": ["fs", "exec", "agent"]}

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

# -----------------------------------------------------------------------------
# Ollama helpers
# -----------------------------------------------------------------------------
def _ollama_chat(payload: Dict[str, Any]) -> requests.Response:
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    return requests.post(url, json=payload, timeout=60)

def _ollama_generate(prompt: str, options: Dict[str, Any]) -> requests.Response:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    body = {"model": MODEL, "prompt": prompt, "stream": False, "options": options or {}}
    return requests.post(url, json=body, timeout=60)

def _extract_answer(resp_json: Dict[str, Any]) -> str:
    # /api/chat: { message: { content: "..." } }
    message = resp_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    # Some models: { messages: [ {content: "..."} ] }
    messages = resp_json.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                text = msg.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    # /api/generate: { response: "..." } or { content/text: "..." }
    for key in ("response", "content", "text"):
        value = resp_json.get(key)
        if isinstance(value, str):
            return value

    return ""

# -----------------------------------------------------------------------------
# Sessions (persist minimal history)
# -----------------------------------------------------------------------------
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
    last_user = next((m.get("content") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), None)
    last_answer = next((m.get("content") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), None)
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        try:
            updated_at = datetime.utcfromtimestamp(_session_path(session_id).stat().st_mtime).isoformat() + "Z"
        except FileNotFoundError:
            updated_at = datetime.utcnow().isoformat() + "Z"
    return {
        "session_id": session_id,
        "turns": len([m for m in messages if isinstance(m, dict) and m.get("role") == "user"]),
        "last_user": last_user,
        "last_answer": last_answer,
        "updated_at": updated_at,
    }

@app.post("/session/new")
def session_new():
    sid = str(uuid.uuid4())
    _save_session(sid, {"messages": []})
    return {"session_id": sid}

@app.delete("/session/{sid}")
def session_delete(sid: str):
    p = _session_path(sid)
    if p.exists():
        p.unlink(missing_ok=True)  # type: ignore[arg-type]
        return {"ok": True, "deleted": sid}
    raise HTTPException(status_code=404, detail=f"Session {sid} not found.")

@app.get("/resume")
def resume(session_id: Optional[str] = Query(None), limit: int = Query(5, ge=1, le=50)):
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

# -----------------------------------------------------------------------------
# Chat (non-stream)
# -----------------------------------------------------------------------------
@app.post("/chat")
def chat(payload: ChatIn = Body(...)):
    """
    Return stable {"answer": "...", "session_id": "..."} responses so CLI/UI stays predictable.
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
        request_body = {"model": MODEL, "messages": history, "stream": bool(payload.stream), "options": options}

        response = _ollama_chat(request_body)

        # Older Ollama builds may not expose /api/chat.
        if response.status_code == 404:
            last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
            response = _ollama_generate(last_user, options)

        if not response.ok:
            try:
                body = response.json()
            except Exception:
                body = response.text[:1000]
            log.error("Ollama upstream error %s: %s", response.status_code, body)
            return {"answer": f"[upstream {response.status_code}] {body}", "session_id": session_id, "messages": history}

        try:
            data = response.json()
        except Exception:
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
        return {"answer": "[timeout] Ollama 60s içinde yanıt vermedi.", "session_id": session_id, "messages": history}
    except requests.RequestException as exc:
        log.exception("Upstream request failed")
        return {"answer": f"[request error] {exc}", "session_id": session_id, "messages": history}
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled server error")
        return {"answer": f"[server error] {exc}", "session_id": session_id, "messages": history}

# -----------------------------------------------------------------------------
# Chat stream (SSE)
# -----------------------------------------------------------------------------
@app.post("/chat/stream")
def chat_stream(payload: ChatIn = Body(...)):
    """
    Server-Sent Events ile token akışı (UI fetch/ReadableStream ile tüketebilir).
    """
    session_id = payload.session_id or str(uuid.uuid4())
    history: List[Dict[str, str]] = []

    if payload.session_id and not payload.reset:
        stored = _load_session(session_id).get("messages", [])
        if isinstance(stored, list):
            history = [m for m in stored if isinstance(m, dict)]
    if payload.reset:
        history = []
    for m in payload.messages:
        history.append({"role": m.role, "content": m.content})

    req = {"model": MODEL, "messages": history, "stream": True, "options": payload.options or {}}
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"

    def gen():
        try:
            with requests.post(url, json=req, stream=True, timeout=300) as r:
                r.raise_for_status()
                buf: List[str] = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        piece = json.loads(line)
                        chunk = _extract_answer(piece)
                        if chunk:
                            buf.append(chunk)
                            yield f"data: {chunk}\n\n"
                    except Exception:
                        continue
                if buf:
                    history.append({"role": "assistant", "content": "".join(buf)})
                    _save_session(session_id, {"messages": history})
                yield f"event: end\ndata: {session_id}\n\n"
        except requests.HTTPError as e:
            yield f"event: error\ndata: upstream {e}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# -----------------------------------------------------------------------------
# FS (repo-scoped)
# -----------------------------------------------------------------------------
def _safe_path(rel: Optional[str]) -> Path:
    if not rel:
        rel = "."
    q = (REPO_ROOT / rel).resolve()
    if not str(q).startswith(str(REPO_ROOT)):
        raise HTTPException(status_code=400, detail="path outside repo root")
    return q

@app.get("/fs/ls", dependencies=[Depends(require_api_key)])
def fs_ls(path: str = Query(".")):
    p = _safe_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    if p.is_file():
        st = p.stat()
        return {"path": str(p.relative_to(REPO_ROOT)), "type": "file", "size": st.st_size, "mtime": st.st_mtime}
    items = []
    for child in sorted(p.iterdir()):
        try:
            st = child.stat()
            items.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                }
            )
        except Exception:
            items.append({"name": child.name, "type": "unknown"})
    return {"path": str(p.relative_to(REPO_ROOT)), "items": items}

@app.get("/fs/read", dependencies=[Depends(require_api_key)])
def fs_read(path: str = Query(...), max_bytes: int = Query(1_000_000, ge=1, le=10_000_000)):
    p = _safe_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    data = p.read_bytes()
    clipped = False
    if len(data) > int(max_bytes):
        data = data[: int(max_bytes)]
        clipped = True
    try:
        text = data.decode("utf-8")
        is_text = True
    except UnicodeDecodeError:
        text = ""
        is_text = False
    return {
        "path": str(p.relative_to(REPO_ROOT)),
        "is_text": is_text,
        "clipped": clipped,
        "size": p.stat().st_size,
        "content": text if is_text else None,
        "hexdump": None if is_text else data.hex(),
    }

@app.post("/fs/write", dependencies=[Depends(require_api_key)])
def fs_write(payload: FSWriteIn):
    p = _safe_path(payload.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload.content, encoding="utf-8")
    return {"ok": True, "path": str(p.relative_to(REPO_ROOT)), "size": len(payload.content)}

# -----------------------------------------------------------------------------
# Exec
# -----------------------------------------------------------------------------
ALLOWED_BIN = {"python", "python3", "pytest", "node", "npm"}
DEFAULT_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "120"))

@app.post("/exec/run", dependencies=[Depends(require_api_key)])
def exec_run(payload: ExecIn):
    if not payload.cmd or not payload.cmd.strip():
        raise HTTPException(status_code=400, detail="empty cmd")
    tokens = shlex.split(payload.cmd)
    first = Path(tokens[0]).name
    if first not in ALLOWED_BIN:
        raise HTTPException(status_code=400, detail=f"command not allowed: {first}")

    cwd = _safe_path(payload.cwd or ".")
    try:
        cp = subprocess.run(
            payload.cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=payload.timeout or DEFAULT_TIMEOUT,
        )
        return {
            "ok": True,
            "cwd": str(cwd.relative_to(REPO_ROOT)),
            "exit_code": cp.returncode,
            "stdout": cp.stdout[-200000:],
            "stderr": cp.stderr[-200000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "cwd": str(cwd.relative_to(REPO_ROOT)),
            "exit_code": None,
            "stdout": (e.stdout or "")[-200000:],
            "stderr": ((e.stderr or "")[-200000:]) + "\n[timeout]",
        }

@app.post("/exec/pytest", dependencies=[Depends(require_api_key)])
def exec_pytest(payload: PytestIn):
    args = ["pytest"]
    if payload.extra:
        args += shlex.split(payload.extra)
    if payload.pattern:
        args.append(payload.pattern)
    cmd = " ".join(args)
    return exec_run(ExecIn(cmd=cmd, cwd=payload.cwd))

# -----------------------------------------------------------------------------
# Agent: Single step
# -----------------------------------------------------------------------------
@app.post("/agent/step", dependencies=[Depends(require_api_key)])
def agent_step(body: AgentStepIn):
    """
    Minimal tool-calling:
      {"action":"fs.read","args":{"path":"api/app.py"}}
      {"action":"fs.write","args":{"path":"demo/a.py","content":"print(1)"}}
      {"action":"exec.run","args":{"cmd":"pytest -q","cwd":"."}}
      {"action":"finish","args":{"result":"..."}}
    """
    try:
        if body.action == "finish":
            return {"done": True, "result": body.args.get("result")}
        if body.action == "fs.read":
            path = body.args.get("path")
            if not isinstance(path, str):
                raise HTTPException(status_code=400, detail="path required")
            return fs_read(path=path, max_bytes=1_000_000)
        if body.action == "fs.write":
            path = body.args.get("path")
            content = body.args.get("content", "")
            if not isinstance(path, str) or not isinstance(content, str):
                raise HTTPException(status_code=400, detail="path/content required")
            return fs_write(FSWriteIn(path=path, content=content))
        if body.action == "exec.run":
            cmd = body.args.get("cmd")
            if not isinstance(cmd, str):
                raise HTTPException(status_code=400, detail="cmd required")
            return exec_run(ExecIn(cmd=cmd, cwd=body.args.get("cwd"), timeout=body.args.get("timeout")))
        raise HTTPException(status_code=400, detail=f"unknown action: {body.action}")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("agent step failed")
        raise HTTPException(status_code=500, detail=f"agent step error: {e}")

# -----------------------------------------------------------------------------
# Agent: Multi-step (ReAct lite)
# -----------------------------------------------------------------------------
def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Find the FIRST valid JSON object in text.
    - Prefer content inside ```json ...``` fences if present
    - If multiple objects appear, return the first successfully decoded dict
    """
    if not isinstance(text, str):
        return None

    # prefer fenced region
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    s = m.group(1) if m else text

    dec = json.JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(s, idx=i)
            if isinstance(obj, dict):
                return obj
            i = end
        except json.JSONDecodeError:
            i += 1
    return None

def _agent_system_prompt() -> str:
    return (
        "You are a local coding agent.\n"
        "TOOLS — return EXACTLY ONE JSON object per turn (no prose):\n"
        '  {"action":"fs.read","args":{"path":"<relpath>"}}\n'
        '  {"action":"fs.write","args":{"path":"<relpath>","content":"<text>"}}\n'
        '  {"action":"exec.run","args":{"cmd":"<command>","cwd":"<dir optional>"}}\n'
        'Finish with: {"action":"finish","args":{"result":"<final answer>"}}\n'
        "Rules:\n"
        "- Return only ONE JSON object each turn.\n"
        "- Use repo-root-relative paths including directories (e.g., 'demo/hello.py').\n"
        "- If you must show JSON inside code fences, ensure the fence wraps ONLY valid JSON.\n"
    )

@app.post("/agent/loop", response_model=AgentLoopOut, dependencies=[Depends(require_api_key)])
def agent_loop(body: AgentLoopIn):
    session_id = body.session_id or str(uuid.uuid4())
    steps: List[AgentLoopStep] = []

    # Build initial messages for the agent
    messages: List[Dict[str, str]] = [{"role": "system", "content": _agent_system_prompt()}]
    messages.append({"role": "user", "content": body.goal})

    for step_idx in range(1, max(1, body.max_steps) + 1):
        # Query model
        req = {"model": MODEL, "messages": messages, "stream": False, "options": {}}
        try:
            resp = _ollama_chat(req)
            if resp.status_code == 404:
                # Fallback to generate with the last user message
                last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
                resp = _ollama_generate(last_user, {})
            resp.raise_for_status()
            data = resp.json()
            answer = (_extract_answer(data) or "").strip()
        except Exception as e:
            steps.append(AgentLoopStep(step=step_idx, error=f"upstream error: {e}"))
            break

        obj = _extract_json_obj(answer)
        if not obj:
            steps.append(AgentLoopStep(step=step_idx, raw=answer, error="no JSON tool call"))
            # add short feedback and continue one more time
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": "Return only ONE JSON object with action/args."})
            continue

        action = obj.get("action")
        args = obj.get("args", {}) if isinstance(obj.get("args"), dict) else {}

        # Execute tool
        observation: Any = None
        error: Optional[str] = None
        try:
            if action == "finish":
                result = args.get("result")
                steps.append(AgentLoopStep(step=step_idx, action=action, args=args, observation=result))
                return AgentLoopOut(session_id=session_id, steps=steps, done=True, result=result)
            elif action == "fs.read":
                p = args.get("path")
                if not isinstance(p, str):
                    raise HTTPException(status_code=400, detail="path required")
                observation = fs_read(path=p, max_bytes=1_000_000)
            elif action == "fs.write":
                p = args.get("path")
                c = args.get("content", "")
                if not isinstance(p, str) or not isinstance(c, str):
                    raise HTTPException(status_code=400, detail="path/content required")
                observation = fs_write(FSWriteIn(path=p, content=c))
            elif action == "exec.run":
                cmd = args.get("cmd")
                if not isinstance(cmd, str):
                    raise HTTPException(status_code=400, detail="cmd required")
                observation = exec_run(ExecIn(cmd=cmd, cwd=args.get("cwd"), timeout=args.get("timeout")))
            else:
                error = f"unknown action: {action}"
        except HTTPException as he:
            error = f"http {he.status_code}: {he.detail}"
        except Exception as e:
            error = f"tool error: {e}"

        steps.append(AgentLoopStep(step=step_idx, action=action, args=args, observation=observation, error=error))

        # Add observation back to the model
        obs_str = json.dumps(observation, ensure_ascii=False) if observation is not None else (error or "")
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"OBSERVATION: {obs_str}\nIf not finished, return the next ONE JSON action."})

    return AgentLoopOut(session_id=session_id, steps=steps, done=False, result=None)

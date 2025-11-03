from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Turkai API — Kod odaklı lokal asistan
# Özellikler:
# - /chat              : Stabil JSON yanıt (Ollama /api/chat -> fallback /api/generate)
# - /chat/stream       : SSE (Server-Sent Events) ile akış
# - /ws/chat           : WebSocket (şimdilik final yanıtı tek mesaj olarak yollar)
# - /session/new|del   : Oturum yönetimi
# - /resume            : Oturum özetleri
# - /fs/*              : Basit repo dosya erişimi (repo kökü dışına çıkmayı engeller)
# - /exec/run|pytest   : Whitelist’li komut çalıştırma (python, pytest vb.)
# - /agent/step        : Mini ReAct — model JSON aksiyon döner, biz uygularız
# - /ui                : Statik arayüz
# Ortam değişkenleri: MODEL, OLLAMA_URL/OLLAMA_HOST, SESSION_DIR, REPO_ROOT, SAFE_ROOT, API_KEY
# ──────────────────────────────────────────────────────────────────────────────

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import json
import uuid
import logging
import re
import requests
import shlex
import subprocess

TOOL_CALL_RE = re.compile(r"<toolcall>\s*(\{.*?\})\s*</toolcall>", re.DOTALL)

from fastapi import (
    FastAPI,
    Body,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Header,
    Depends,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger("turkai")
logging.basicConfig(level=logging.INFO)

# ── Config
MODEL = os.getenv("MODEL") or os.getenv("MODEL_NAME") or "qwen2.5-coder:1.5b"
OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "http://ollama:11434"
SESSION_DIR = Path(os.getenv("SESSION_DIR", "./data/sessions")).resolve()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
API_KEY = os.getenv("API_KEY", "")  # boş ise auth kapalı

# Repo kökü ve güvenli kök
REPO_ROOT = Path(os.getenv("REPO_ROOT", ".")).resolve()
SAFE_ROOT = Path(os.getenv("SAFE_ROOT", str(REPO_ROOT))).resolve()
DEFAULT_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "120"))
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    (
        "You are Turkai, a local coding assistant operating on the repository mounted at "
        f"{SAFE_ROOT}. You must interact with that filesystem via tool calls in the form "
        '<toolcall>{"tool": "...", "args": {...}}</toolcall>. Available tools:\n'
        "- context.tree(path=\".\", max_depth=2, max_entries=200)        # enumerate dirs/files\n"
        "- context.snippet(path, line=1, context=8)                      # read file excerpts (files only)\n"
        "- fs.search(pattern, path=\".\", max_results=50, case_sensitive=False)\n"
        "- fs.write(path, content)                                       # create or overwrite files\n"
        "- context.diff(paths=None, staged=False, unified=3)             # review git diff\n"
        "- exec.run(cmd, cwd=\".\")                                       # run commands/tests\n"
        "Each turn follow this protocol: begin with 'Plan:' and concise reasoning, then immediately emit"
        " the next tool call that advances the plan. If a tool response contains an error or \"ok\": false"
        " (e.g. path missing, directory passed to snippet), issue another tool call to recover before replying."
        " Use context.tree before assuming paths and use fs.write—never snippet—to create or modify files."
        " Once finished, reply in natural language, summarising changes and suggesting next steps if relevant."
    ),
)

SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Pydantic modelleri
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
    extra: Optional[str] = None  # örn: "-k quick -q"


class AgentIn(BaseModel):
    session_id: Optional[str] = None
    messages: List[Message]
    allow: Optional[List[str]] = None  # e.g. ["fs.read","fs.write","exec.run"]
    cwd: Optional[str] = "."


class TreeIn(BaseModel):
    path: Optional[str] = "."
    max_depth: int = 2
    max_entries: int = 400


class SnippetIn(BaseModel):
    path: str
    line: int = 1
    context: int = 8


class SearchIn(BaseModel):
    pattern: str
    path: Optional[str] = "."
    max_results: int = 50
    case_sensitive: bool = False


class DiffIn(BaseModel):
    paths: Optional[List[str]] = None
    staged: bool = False
    unified: int = 3


# ── FastAPI app
app = FastAPI(title="Turkai API", version="0.2")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Yardımcılar
def require_api_key(x_api_key: str | None = Header(default=None)) -> bool:
    """Opsiyonel API-Key koruması. API_KEY boşsa dev modda kapalıdır."""
    if not API_KEY:
        return True
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    return True


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

    # Bazı modeller messages döndürebilir
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

    # Ollama /api/generate
    for key in ("response", "content", "text"):
        value = resp_json.get(key)
        if isinstance(value, str):
            return value

    return ""


def _ensure_system_prompt(history: List[Dict[str, str]], incoming: List[Message]) -> None:
    has_system = any(m.get("role") == "system" for m in history)
    incoming_has_system = any(m.role == "system" for m in incoming)
    if not has_system and not incoming_has_system:
        history.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})


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


def _safe_path(rel: str | None, base: Path | None = None) -> Path:
    """REPO_ROOT/SAFE_ROOT dışına çıkmayı engeller."""
    if not rel:
        rel = "."
    base_dir = (base or REPO_ROOT).resolve()
    q = (base_dir / rel).resolve()
    if not str(q).startswith(str(SAFE_ROOT)):
        raise HTTPException(status_code=400, detail=f"path outside safe root: {q}")
    return q


def _exec_run(cmd: str, cwd: str | None = ".", timeout: int = 120) -> Dict[str, Any]:
    """shell=False + shlex.split ile güvenli çalıştırma; çıktı/timeout döndürür."""
    try:
        args = shlex.split(cmd)
    except Exception as e:
        return {"ok": False, "error": f"bad command line: {e}"}
    try:
        p = subprocess.run(
            args,
            cwd=_safe_path(cwd).as_posix(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": True,
            "exit_code": p.returncode,
            "stdout": p.stdout[-20000:],
            "stderr": p.stderr[-20000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _iter_tree(root: Path, max_depth: int, max_entries: int) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    queue: List[tuple[Path, int]] = [(root, 0)]
    while queue and len(entries) < max_entries:
        current, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except Exception:
            continue
        for child in children:
            try:
                rel = child.relative_to(REPO_ROOT)
            except ValueError:
                continue
            info = {
                "path": str(rel),
                "type": "dir" if child.is_dir() else "file",
                "depth": depth,
            }
            if child.is_file():
                try:
                    info["size"] = child.stat().st_size
                except Exception:
                    info["size"] = None
            entries.append(info)
            if len(entries) >= max_entries:
                break
            if child.is_dir():
                queue.append((child, depth + 1))
    return entries[:max_entries]


def _read_snippet(path: Path, line: int, context: int) -> Dict[str, Any]:
    if line < 1:
        line = 1
    if context < 0:
        context = 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    total = len(lines)
    start = max(1, line - context)
    end = min(total, line + context)
    snippet = "\n".join(lines[start - 1 : end])
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "total_lines": total,
        "start_line": start,
        "end_line": end,
        "snippet": snippet,
    }


def _search_pattern(
    root: Path,
    pattern: str,
    max_results: int,
    case_sensitive: bool,
    only_files: Optional[List[Path]] = None,
) -> List[Dict[str, Any]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise HTTPException(status_code=400, detail=f"invalid regex: {exc}") from exc

    matches: List[Dict[str, Any]] = []
    candidates: List[Path]
    if only_files:
        candidates = only_files
    else:
        candidates = []
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                candidates.append(Path(dirpath) / name)
                if len(candidates) >= max_results * 20:
                    break
            if len(candidates) >= max_results * 20:
                break
    for p in candidates:
        if len(matches) >= max_results:
            break
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        try:
            data = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for idx, line in enumerate(data.splitlines(), start=1):
            if regex.search(line):
                matches.append({"path": str(rel), "line": idx, "preview": line.strip()})
                if len(matches) >= max_results:
                    break
    return matches


def _execute_toolcall(tool: str, args: Dict[str, Any]) -> str:
    try:
        if tool == "context.tree":
            path = args.get("path", ".")
            depth = int(args.get("max_depth", 2))
            entries = _iter_tree(_safe_path(path), max_depth=depth, max_entries=int(args.get("max_entries", 200)))
            return json.dumps({"entries": entries}, ensure_ascii=False)[:8000]

        if tool == "context.snippet":
            path = args.get("path")
            if not isinstance(path, str):
                raise ValueError("snippet requires 'path'")
            line = int(args.get("line", 1))
            context = int(args.get("context", 8))
            data = _read_snippet(_safe_path(path), line, context)
            return json.dumps(data, ensure_ascii=False)[:8000]

        if tool == "fs.search":
            pattern = args.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError("search requires 'pattern'")
            path = args.get("path", ".")
            matches = _search_pattern(
                _safe_path(path),
                pattern,
                int(args.get("max_results", 50)),
                bool(args.get("case_sensitive", False)),
            )
            return json.dumps({"matches": matches}, ensure_ascii=False)[:8000]

        if tool == "fs.write":
            path = args.get("path")
            content = args.get("content", "")
            if not isinstance(path, str):
                raise ValueError("fs.write requires 'path'")
            p = _safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(content), encoding="utf-8")
            return json.dumps({"written": len(str(content)), "path": str(p.relative_to(REPO_ROOT))})

        if tool == "context.diff":
            rel_paths = []
            for p in args.get("paths") or []:
                rel_paths.append(str(_safe_path(p).relative_to(REPO_ROOT)))
            data = _git_diff(rel_paths, bool(args.get("staged", False)), int(args.get("unified", 3)))
            return json.dumps(data, ensure_ascii=False)[:8000]

        if tool == "exec.run":
            payload = ExecIn(**args)
            data = exec_run(payload)  # type: ignore[arg-type]
            return json.dumps(data, ensure_ascii=False)[:8000]

        raise ValueError(f"unknown tool: {tool}")
    except HTTPException as exc:
        return json.dumps({"ok": False, "error": exc.detail}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


# ── Kök/sağlık/UI
@app.get("/")
def root():
    return {
        "name": "Turkai API",
        "model": MODEL,
        "tools_enabled": True,
        "tools": ["fs", "exec", "agent", "context"],
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/ui", include_in_schema=False)
def ui_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not bundled</h1>", status_code=404)


# ── Chat (stabil JSON)
@app.post("/chat")
def chat(payload: ChatIn = Body(...)):
    """
    Stabil {"answer": "...", "session_id": "..."} döndürür; CLI/WS/UI için güvenilir.
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

        _ensure_system_prompt(history, payload.messages)

        for message in payload.messages:
            history.append({"role": message.role, "content": message.content})

        options = payload.options or {}

        final_answer: Optional[str] = None
        for _ in range(6):
            request_body = {
                "model": MODEL,
                "messages": history,
                "stream": False,
                "options": options,
            }

            response = _ollama_chat(request_body)

            if response.status_code == 404:
                last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
                response = _ollama_generate(last_user, options)

            if not response.ok:
                try:
                    body = response.json()
                except Exception:
                    body = response.text[:1000]
                log.error("Ollama upstream error %s: %s", response.status_code, body)
                return {
                    "answer": f"[upstream {response.status_code}] {body}",
                    "session_id": session_id,
                    "messages": history,
                }

            try:
                data = response.json()
            except Exception:
                text = response.text if response.text else "[upstream returned non-JSON]"
                return {"answer": text[:1000], "session_id": session_id, "messages": history}

            answer = (_extract_answer(data) or "").strip()
            if not answer:
                return {"answer": "[empty answer]", "session_id": session_id, "messages": history}

            matches = TOOL_CALL_RE.findall(answer)
            if not matches:
                final_answer = answer
                history.append({"role": "assistant", "content": answer})
                break

            history.append({"role": "assistant", "content": answer})
            for match in matches:
                tool_name = "unknown"
                try:
                    call = json.loads(match)
                    tool = call.get("tool")
                    args = call.get("args", {})
                    if not isinstance(tool, str):
                        raise ValueError("missing tool name")
                    tool_name = tool
                    if not isinstance(args, dict):
                        raise ValueError("args must be object")
                    observation = _execute_toolcall(tool, args)
                except Exception as exc:  # noqa: BLE001
                    observation = json.dumps({"ok": False, "error": str(exc)})
                history.append({"role": "user", "content": f"[tool:{tool_name}] {observation}"})

        if final_answer is None:
            final_answer = "[warning] tool loop exhausted"

        _save_session(session_id, {"messages": history})
        return {"answer": final_answer, "session_id": session_id, "messages": history}

    except requests.Timeout:
        log.exception("Upstream timeout")
        return {"answer": "[timeout] Ollama 60s içinde yanıt vermedi.", "session_id": session_id, "messages": history}
    except requests.RequestException as exc:
        log.exception("Upstream request failed")
        return {"answer": f"[request error] {exc}", "session_id": session_id, "messages": history}
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled server error")
        return {"answer": f"[server error] {exc}", "session_id": session_id, "messages": history}


# ── SSE Stream
@app.post("/chat/stream")
def chat_stream(payload: ChatIn = Body(...)):
    session_id = payload.session_id or str(uuid.uuid4())
    history: List[Dict[str, str]] = []
    if payload.session_id and not payload.reset:
        stored = _load_session(session_id).get("messages", [])
        if isinstance(stored, list):
            history = [m for m in stored if isinstance(m, dict)]
    if payload.reset:
        history = []
    _ensure_system_prompt(history, payload.messages)
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


# ── WebSocket (tek yanıt)
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        first = await ws.receive_text()
        try:
            req = json.loads(first)
        except Exception:
            await ws.send_text(json.dumps({"error": "invalid json"}))
            await ws.close()
            return
        payload = ChatIn(**req)
        resp = chat(payload)  # type: ignore
        await ws.send_text(json.dumps(resp))
        await ws.close()
    except WebSocketDisconnect:
        pass


# ── Session helpers
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


# ── FS uçları
@app.get("/fs/ls")
def fs_ls(path: str = Query(".")):
    p = _safe_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    if p.is_file():
        stat = p.stat()
        return {
            "path": str(p.relative_to(REPO_ROOT)),
            "type": "file",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
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


@app.get("/fs/read")
def fs_read(path: str = Query(...), max_bytes: int = Query(1_000_000, ge=1, le=10_000_000)):
    p = _safe_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    data = p.read_bytes()
    clipped = False
    if len(data) > max_bytes:
        data = data[:max_bytes]
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


@app.post("/fs/write")
def fs_write(payload: FSWriteIn):
    p = _safe_path(payload.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload.content, encoding="utf-8")
    return {"ok": True, "path": str(p.relative_to(REPO_ROOT)), "size": len(payload.content)}


@app.post("/context/tree")
def context_tree(payload: TreeIn = Body(...)):
    root = _safe_path(payload.path)
    if not root.exists():
        raise HTTPException(status_code=404, detail="path not found")
    if not root.is_dir():
        root = root.parent
    entries = _iter_tree(root, max_depth=payload.max_depth, max_entries=payload.max_entries)
    return {"root": str(root.relative_to(REPO_ROOT)), "entries": entries}


@app.post("/context/snippet")
def context_snippet(payload: SnippetIn = Body(...)):
    p = _safe_path(payload.path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return _read_snippet(p, payload.line, payload.context)


def _git_diff(rel_paths: List[str], staged: bool, unified: int) -> Dict[str, Any]:
    args = [
        "git",
        "-C",
        str(REPO_ROOT),
        "diff",
        "--color=never",
        f"-U{max(0, unified)}",
    ]
    if staged:
        args.append("--staged")
    if rel_paths:
        args.append("--")
        args.extend(rel_paths)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
        )
    except FileNotFoundError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="git binary not available") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="git diff timeout") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if proc.returncode not in (0, 1):
        raise HTTPException(status_code=500, detail=proc.stderr.strip() or "git diff failed")

    return {"diff": proc.stdout, "paths": rel_paths, "staged": staged}


@app.post("/fs/search")
def fs_search(payload: SearchIn = Body(...)):
    root = _safe_path(payload.path)
    files: Optional[List[Path]] = None
    if root.is_file():
        base = root.parent
        files = [root]
    else:
        base = root
    matches = _search_pattern(base, payload.pattern, payload.max_results, payload.case_sensitive, files)
    return {"matches": matches, "root": str(base.relative_to(REPO_ROOT))}


@app.post("/context/diff")
def context_diff(payload: DiffIn = Body(...)):
    rel_paths: List[str] = []
    for p in payload.paths or []:
        safe = _safe_path(p)
        rel_paths.append(str(safe.relative_to(REPO_ROOT)))
    return _git_diff(rel_paths, payload.staged, payload.unified)


# ── Exec uçları
ALLOWED_BIN = {"python", "python3", "pytest", "node", "npm"}  # gerekirse genişlet


@app.post("/exec/run")
def exec_run(payload: ExecIn):
    if not payload.cmd or not payload.cmd.strip():
        raise HTTPException(status_code=400, detail="empty cmd")
    # İlk token whitelist’te mi?
    first = Path(shlex.split(payload.cmd)[0]).name
    if first not in ALLOWED_BIN:
        raise HTTPException(status_code=400, detail=f"command not allowed: {first}")

    res = _exec_run(payload.cmd, cwd=payload.cwd or ".", timeout=payload.timeout or DEFAULT_TIMEOUT)
    # Çalışma dizinini ekleyelim
    res["cwd"] = str(_safe_path(payload.cwd or ".").relative_to(REPO_ROOT))
    return res


@app.post("/exec/pytest")
def exec_pytest(payload: PytestIn):
    args = ["pytest"]
    if payload.extra:
        args += shlex.split(payload.extra)
    if payload.pattern:
        args.append(payload.pattern)
    cmd = " ".join(args)
    return exec_run(ExecIn(cmd=cmd, cwd=payload.cwd))


# ── Mini Tool-Calling (ReAct-lite)
SYS_PROMPT_REACT = """You are a coding agent. Think step-by-step, but RESPOND ONLY with compact JSON.
Either propose an ACTION or produce FINAL result.
Schema:
  {"action":"fs.read","params":{"path":"..."}}
  {"action":"fs.write","params":{"path":"...","content":"..."}}
  {"action":"exec.run","params":{"cmd":"..."}}
  {"final":"answer text"}
DO NOT include markdown, just minified JSON."""


@app.post("/agent/step")
def agent_step(payload: AgentIn = Body(...), _: bool = Depends(require_api_key)):
    allow = set(payload.allow or [])

    # 1) Modelden JSON aksiyon iste
    req_body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS_PROMPT_REACT}] + [m.model_dump() for m in payload.messages],
        "stream": False,
        "options": {},
    }
    r = _ollama_chat(req_body)
    if not r.ok:
        return {"ok": False, "error": f"upstream {r.status_code}"}
    try:
        ans = r.json()
        text = _extract_answer(ans)
        cmd = json.loads(text)
    except Exception as e:
        return {"ok": False, "error": f"invalid model JSON: {e}", "raw": r.text[:1000]}

    # 2) Final ise bitir
    if isinstance(cmd, dict) and "final" in cmd:
        return {"ok": True, "final": cmd["final"]}

    # 3) Aksiyon uygula
    if not isinstance(cmd, dict) or "action" not in cmd or "params" not in cmd:
        return {"ok": False, "error": "bad action schema", "cmd": cmd}
    action = str(cmd["action"])
    params = cmd["params"] or {}
    cwd = payload.cwd or "."

    if action == "fs.read":
        if "fs.read" not in allow:
            return {"ok": False, "error": "fs.read not allowed"}
        p = _safe_path(params.get("path", ""), base=_safe_path(cwd))
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "observation": {"content": content[-20000:], "path": str(p)}}

    if action == "fs.write":
        if "fs.write" not in allow:
            return {"ok": False, "error": "fs.write not allowed"}
        p = _safe_path(params.get("path", ""), base=_safe_path(cwd))
        content = params.get("content", "")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "observation": {"written": len(content), "path": str(p)}}

    if action == "exec.run":
        if "exec.run" not in allow:
            return {"ok": False, "error": "exec.run not allowed"}
        cmdline = params.get("cmd", "")
        res = _exec_run(cmdline, cwd=cwd, timeout=DEFAULT_TIMEOUT)
        return {"ok": True, "observation": res}

    return {"ok": False, "error": f"unknown action: {action}", "cmd": cmd}

# ==== Mini tool-calling: /agent/step ==========================================
from typing import Literal

class AgentStepIn(BaseModel):
    session_id: Optional[str] = None
    action: Literal["fs.read", "fs.write", "exec.run"]
    args: Dict[str, Any] = {}

@app.post("/agent/step")
def agent_step(body: AgentStepIn):
    """
    Basit tek-adım tool-calling:
      {"action":"fs.read","args":{"path":"api/app.py"}}
      {"action":"fs.write","args":{"path":"demo/a.py","content":"print(1)"}}
      {"action":"exec.run","args":{"cmd":"pytest -q","cwd":"."}}
    """
    try:
        if body.action == "fs.read":
            path = body.args.get("path")
            if not isinstance(path, str):
                raise HTTPException(status_code=400, detail="path required")
            return fs_read(path=path)

        elif body.action == "fs.write":
            path = body.args.get("path")
            content = body.args.get("content", "")
            if not isinstance(path, str) or not isinstance(content, str):
                raise HTTPException(status_code=400, detail="path/content required")
            return fs_write(FSWriteIn(path=path, content=content))

        elif body.action == "exec.run":
            cmd = body.args.get("cmd")
            if not isinstance(cmd, str):
                raise HTTPException(status_code=400, detail="cmd required")
            return exec_run(ExecIn(cmd=cmd, cwd=body.args.get("cwd"), timeout=body.args.get("timeout")))

        else:
            raise HTTPException(status_code=400, detail=f"unknown action: {body.action}")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("agent step failed")
        raise HTTPException(status_code=500, detail=f"agent step error: {e}")

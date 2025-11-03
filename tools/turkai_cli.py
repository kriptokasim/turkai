#!/usr/bin/env python3

"""
Minimal diagnostic CLI for Turkai.

Allows you to:
  - send chat messages (tracks session automatically)
  - resume existing sessions
  - inspect repo context via tree/snippet/search/diff helpers
  - exercise the agent loop for Codex-style operations

Commands (prefixed with ':'):
  :resume <session_id>      -> switch to an existing session (loads history)
  :new                      -> start a new session
  :history                  -> show the last messages for the active session
  :tree [path] [depth]      -> call /context/tree
  :snippet <path> [line ctx]-> call /context/snippet
  :search <regex> [path]    -> call /fs/search
  :diff [--staged] [paths]  -> call /context/diff
  :agent <allow...>         -> run /agent/step with the last chat
  :quit / :exit             -> leave the CLI

Usage:
  python tools/turkai_cli.py
  python tools/turkai_cli.py --url http://localhost:8011 --resume SESSION_ID
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import requests


def pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "TURKAI_SYSTEM_PROMPT",
    (
        "You are Turkai, a local coding assistant with direct access to the project repository via the "
        "Turkai tooling API. Consult /context/tree, /context/snippet, /fs/search, /context/diff, and "
        "/agent/step to inspect and modify files or execute commands. Think step by step and describe your "
        "plan before applying edits."
    ),
)


@dataclass
class TurkaiClient:
    base_url: str
    api_key: Optional[str] = None
    session_id: Optional[str] = None
    history: List[dict] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    system_injected: bool = False

    # ----- HTTP helpers -----
    def _headers(self) -> dict:
        hdrs = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["X-API-Key"] = self.api_key
        return hdrs

    def _post(self, path: str, payload: dict) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}" + path
        return requests.post(url, headers=self._headers(), json=payload, timeout=60)

    def _get(self, path: str, params: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}" + path
        return requests.get(url, headers=self._headers(), params=params, timeout=60)

    # ----- Chat -----
    def send_chat(self, text: str) -> str:
        msg = {"role": "user", "content": text}
        messages: List[dict] = []
        if not self.system_injected:
            messages.append({"role": "system", "content": self.system_prompt})
            self.system_injected = True
        payload = {
            "messages": messages + [msg],
            "session_id": self.session_id,
        }
        resp = self._post("/chat", payload)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id") or self.session_id
        self.history = data.get("messages") or []
        return data.get("answer", "")

    # ----- Resume -----
    def resume(self, session_id: str) -> List[dict]:
        resp = self._get("/resume", {"session_id": session_id})
        resp.raise_for_status()
        data = resp.json()
        self.session_id = session_id
        self.history = data.get("messages") or []
        self.system_injected = any(m.get("role") == "system" for m in self.history)
        return self.history

    # ----- Context helpers -----
    def tree(self, path: str = ".", depth: int = 2) -> dict:
        resp = self._post("/context/tree", {"path": path, "max_depth": depth, "max_entries": 200})
        resp.raise_for_status()
        return resp.json()

    def snippet(self, path: str, line: int = 1, context: int = 8) -> dict:
        resp = self._post("/context/snippet", {"path": path, "line": line, "context": context})
        resp.raise_for_status()
        return resp.json()

    def search(self, pattern: str, path: str = ".", max_results: int = 20, case_sensitive: bool = False) -> dict:
        payload = {
            "pattern": pattern,
            "path": path,
            "max_results": max_results,
            "case_sensitive": case_sensitive,
        }
        resp = self._post("/fs/search", payload)
        resp.raise_for_status()
        return resp.json()

    def diff(self, paths: List[str], staged: bool = False, unified: int = 3) -> dict:
        payload = {
            "paths": paths or None,
            "staged": staged,
            "unified": unified,
        }
        resp = self._post("/context/diff", payload)
        resp.raise_for_status()
        return resp.json()

    def agent_step(self, allow: List[str], cwd: str = ".") -> dict:
        if not self.history:
            raise RuntimeError("No conversation history to feed into agent.")
        payload = {
            "messages": self.history[-6:],
            "allow": allow,
            "cwd": cwd,
        }
        resp = self._post("/agent/step", payload)
        resp.raise_for_status()
        return resp.json()


def repl(client: TurkaiClient) -> None:
    print(f"Connected to Turkai at {client.base_url}")
    if client.session_id:
        print(f"Resumed session {client.session_id}")
    print("Type ':help' for commands.")
    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            try:
                handle_command(client, line[1:])
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[error] {exc}")
            continue
        try:
            answer = client.send_chat(line)
            print(f"[assistant] {answer}")
        except requests.HTTPError as exc:
            print(f"[error] chat failed: {exc} -> {exc.response.text[:200]}")


def handle_command(client: TurkaiClient, cmdline: str) -> None:
    parts = shlex.split(cmdline)
    if not parts:
        return
    cmd, *args = parts

    if cmd in {"quit", "exit"}:
        raise KeyboardInterrupt
    if cmd == "help":
        print(__doc__.strip())
        return
    if cmd == "resume":
        if not args:
            print("[error] usage: :resume <session_id>")
            return
        hist = client.resume(args[0])
        print(f"[info] resumed {args[0]} ({len(hist)} messages)")
        for msg in hist[-8:]:
            print(f"[{msg.get('role')}] {msg.get('content')}")
        return
    if cmd == "history":
        if not client.history:
            print("[info] no history yet")
            return
        for msg in client.history[-12:]:
            print(f"[{msg.get('role')}] {msg.get('content')}")
        return
    if cmd == "new":
        client.session_id = None
        client.history = []
        client.system_injected = False
        print("[info] started new session")
        return
    if cmd == "tree":
        path = args[0] if args else "."
        depth = int(args[1]) if len(args) > 1 else 2
        res = client.tree(path, depth)
        print(pretty(res))
        return
    if cmd == "snippet":
        if not args:
            print("[error] usage: :snippet path [line] [context]")
            return
        path = args[0]
        line = int(args[1]) if len(args) > 1 else 1
        ctx = int(args[2]) if len(args) > 2 else 8
        res = client.snippet(path, line, ctx)
        print(pretty(res))
        if res.get("snippet"):
            print(res["snippet"])
        return
    if cmd == "search":
        if not args:
            print("[error] usage: :search <pattern> [path]")
            return
        pattern = args[0]
        path = args[1] if len(args) > 1 else "."
        res = client.search(pattern, path)
        print(pretty(res))
        return
    if cmd == "diff":
        staged = False
        unified = 3
        paths: List[str] = []
        for arg in args:
            if arg == "--staged":
                staged = True
            elif arg.startswith("-U"):
                unified = int(arg[2:] or "3")
            else:
                paths.append(arg)
        res = client.diff(paths, staged, unified)
        print(res.get("diff") or "[no diff]")
        return
    if cmd == "agent":
        allow = args or ["fs.read", "fs.write", "exec.run"]
        res = client.agent_step(allow)
        print(pretty(res))
        return

    print(f"[error] unknown command: {cmd}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Turkai debug CLI")
    parser.add_argument("--url", default=os.environ.get("TURKAI_URL", "http://127.0.0.1:8011"))
    parser.add_argument("--api-key", default=os.environ.get("TURKAI_API_KEY"))
    parser.add_argument("--resume")
    parser.add_argument("--system", help="Override default system prompt")
    args = parser.parse_args(argv)

    client = TurkaiClient(base_url=args.url, api_key=args.api_key)
    if args.system:
        client.system_prompt = args.system
        client.system_injected = False
    if args.resume:
        try:
            client.resume(args.resume)
        except requests.HTTPError as exc:
            print(f"[error] resume failed: {exc}")
    try:
        repl(client)
    except KeyboardInterrupt:
        print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

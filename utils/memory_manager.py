from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional

from config import CHAT_DIR, SETTINGS_FILE


def ensure_chat_dir() -> None:
    os.makedirs(CHAT_DIR, exist_ok=True)


def _session_path(chat_id: str) -> str:
    return os.path.join(CHAT_DIR, f"{chat_id}.json")


def new_session() -> Dict:
    ensure_chat_dir()
    chat_id = str(uuid.uuid4())
    now = time.time()
    session = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_session(session)
    return session


def load_session(chat_id: str) -> Optional[Dict]:
    path = _session_path(chat_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_session(session: Dict) -> str:
    ensure_chat_dir()
    session["updated_at"] = time.time()
    path = _session_path(session["id"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, ensure_ascii=False, indent=2)
    return path


def list_sessions() -> List[Dict]:
    ensure_chat_dir()
    sessions: List[Dict] = []
    settings_exists = os.path.exists(SETTINGS_FILE)
    for filename in os.listdir(CHAT_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(CHAT_DIR, filename)
        if settings_exists and os.path.samefile(path, SETTINGS_FILE):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            session = json.load(handle)
            sessions.append(session)
    sessions.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return sessions


def update_title_if_needed(session: Dict, user_message: str) -> None:
    if session.get("title") != "New Chat":
        return
    title = user_message.strip().splitlines()[0] if user_message.strip() else "New Chat"
    session["title"] = title[:60]


def clear_session(session: Dict) -> Dict:
    session["messages"] = []
    session["title"] = "New Chat"
    save_session(session)
    return session


def export_session(session: Dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(session, ensure_ascii=False, indent=2)
    if fmt == "markdown":
        lines: List[str] = [f"# {session.get('title', 'Chat')}", ""]
        for message in session.get("messages", []):
            role = message.get("role", "assistant").title()
            content = message.get("content", "")
            lines.append(f"## {role}")
            lines.append("")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()
    raise ValueError(f"Unsupported export format: {fmt}")


def load_settings() -> Dict:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_settings(settings: Dict) -> None:
    ensure_chat_dir()
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
"""Долговременная память агента: история всех диалогов в memory.json.

Чем отличается от короткой памяти из прошлого урока: там история жила в оперативной
памяти процесса и исчезала вместе с терминалом. Здесь она лежит на диске, поэтому
новый запуск агента продолжает старый разговор.

Формат файла:
    {
      "version": 1,
      "created_at": "2026-10-15T12:00:00",
      "updated_at": "2026-10-15T12:07:41",
      "sessions": [
        {
          "id": 1,
          "started_at": "...", "ended_at": "...",
          "turns": 2,
          "messages": [
            {"role": "user", "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...",
             "timestamp": "...", "tools_used": ["get_weather"]}
          ]
        }
      ]
    }

В модель уходит не весь файл, а последние CONTEXT_MESSAGES сообщений из последней
сессии — иначе контекст быстро переполнится и запрос станет дорогим.

Запись атомарная: сначала рядом пишется временный файл, потом он заменяет основной
через os.replace. Так внезапное закрытие терминала не оставит обрезанный memory.json.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"
SCHEMA_VERSION = 1
CONTEXT_MESSAGES = 20  # столько последних сообщений агент показывает модели
MAX_STORED_CHARS = 4000  # длинный ответ модели в память не пишется целиком


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Memory:
    """История диалогов в memory.json: чтение при старте, дозапись после каждого ответа."""

    def __init__(self, path: Path | str = MEMORY_FILE, context_messages: int = CONTEXT_MESSAGES) -> None:
        self.path = Path(path)
        self.context_messages = context_messages
        self.data = self._load()
        self.session = self._open_session()

    # --- чтение и запись файла ------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        """Читает memory.json; при первом запуске или битом файле начинает с чистого листа."""
        empty = {
            "version": SCHEMA_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "sessions": [],
        }
        if not self.path.exists():
            return empty
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # Испорченный файл не должен ронять агента: он переименовывается
            # в .broken, чтобы данные можно было посмотреть вручную.
            try:
                self.path.replace(self.path.with_suffix(".json.broken"))
            except OSError:
                pass
            return empty

        if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
            return empty
        data.setdefault("version", SCHEMA_VERSION)
        data.setdefault("created_at", _now())
        return data

    def save(self) -> None:
        """Атомарно перезаписывает memory.json."""
        self.data["version"] = SCHEMA_VERSION
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix="memory_",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    # --- сессии ---------------------------------------------------------------

    def _open_session(self) -> Dict[str, Any]:
        """Продолжает последнюю сессию, если она сегодняшняя, иначе начинает новую."""
        sessions = self.data["sessions"]
        today = datetime.now().date().isoformat()
        if sessions and str(sessions[-1].get("started_at", ""))[:10] == today:
            return sessions[-1]

        session = {
            "id": len(sessions) + 1,
            "started_at": _now(),
            "ended_at": None,
            "turns": 0,
            "messages": [],
        }
        sessions.append(session)
        return session

    # --- сообщения ------------------------------------------------------------

    def add(self, role: str, content: str, tools_used: Optional[List[str]] = None) -> None:
        """Дописывает сообщение в текущую сессию."""
        content = (content or "").strip()
        if not content:
            return

        message: Dict[str, Any] = {
            "role": role,
            "content": content[:MAX_STORED_CHARS],
            "timestamp": _now(),
        }
        if tools_used:
            message["tools_used"] = tools_used
        self.session["messages"].append(message)

        if role == "assistant":
            self.session["turns"] = self.session.get("turns", 0) + 1
        self.save()

    def history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Последние сообщения в формате, который понимает Chat Completions."""
        window = self.context_messages if limit is None else limit
        messages = self.session.get("messages", [])
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages[-window:]
        ]

    # --- статистика и управление ----------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Сводка для команды /stats в CLI и для README при отладке."""
        sessions = self.data["sessions"]
        return {
            "file": str(self.path),
            "exists": self.path.exists(),
            "sessions": len(sessions),
            "messages": sum(len(s.get("messages", [])) for s in sessions),
            "current_session": self.session["id"],
            "turns": self.session.get("turns", 0),
            "in_context": len(self.history()),
        }

    def clear(self) -> None:
        """Забывает всё: файл перезаписывается пустой историей."""
        self.data["sessions"] = []
        self.data["created_at"] = _now()
        self.session = self._open_session()
        self.save()
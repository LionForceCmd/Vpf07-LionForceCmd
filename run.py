"""Точка входа: терминальный интерфейс агента.

Запуск:
    python run.py                 # диалог в консоли
    python run.py "погода в Казани"  # один запрос и выход
    python run.py --log-info      # подробный лог в консоль (в agent.log он всегда)

Логирование сделано по шагам цикла агента. Именно оно в уроке показало, где ломается
агент: без него ошибка «File not found» выглядит как «модель неправильно поняла задачу».
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent import dump_transcript, greet, run
from llm import is_offline
from memory import Memory
from tools import TOOL_NAMES

load_dotenv()

LOG_FILE = Path(__file__).resolve().parent / "agent.log"


def setup_logging(verbose: bool = False) -> None:
    """Пишет лог и в файл, и в консоль; консольный уровень зависит от --log-info."""
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.basicConfig(level=logging.INFO, handlers=[], force=True)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger().addHandler(console_handler)


COMMANDS = {
    "/exit": "выйти из агента",
    "/quit": "выйти из агента",
    "/help": "эта справка",
    "/tools": "список инструментов",
    "/stats": "состояние памяти memory.json",
    "/memory": "последние сообщения из памяти",
    "/clear": "очистить всю историю диалога",
    "/transcript": "выгрузить диалог последнего цикла в transcript.json",
}


def handle_command(text: str, memory: Memory, state: dict) -> bool:
    """Обрабатывает команду. Возвращает False, если пора выходить."""
    command = text.strip().lower()

    if command in ("/exit", "/quit"):
        print("Пока.")
        return False

    if command == "/help":
        width = max(len(name) for name in COMMANDS)
        print("\n".join(f"  {name:<{width}} — {COMMANDS[name]}" for name in COMMANDS))
        return True

    if command == "/tools":
        print("\n".join(f"  {i + 1}. {name}" for i, name in enumerate(TOOL_NAMES)))
        return True

    if command == "/stats":
        for key, value in memory.stats().items():
            print(f"  {key}: {value}")
        return True

    if command == "/memory":
        for message in memory.history(limit=6):
            print(f'  [{message["role"]}] {message["content"][:160]}')
        return True

    if command == "/clear":
        memory.clear()
        print("История диалога очищена.")
        return True

    if command == "/transcript":
        path = Path(__file__).resolve().parent / "transcript.json"
        path.write_text(dump_transcript(state.get("messages", [])), encoding="utf-8")
        print(f"Диалог последнего цикла сохранён в {path}")
        return True

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Терминальный AI-агент")
    parser.add_argument("query", nargs="*", help="один запрос; без него — диалог в консоли")
    parser.add_argument("--log-info", action="store_true", help="подробный лог в консоль")
    parser.add_argument("--memory", default=None, help="путь к файлу памяти (по умолчанию memory.json)")
    args = parser.parse_args()

    setup_logging(args.log_info)
    log = logging.getLogger("run")

    memory = Memory(path=args.memory) if args.memory else Memory()
    state: dict = {}

    log.info(
        "старт: режим=%s, память=%s",
        "offline" if is_offline() else "openai",
        memory.path,
    )

    single = " ".join(args.query).strip()
    if single:
        print(run(single, memory, state))
        return

    print(greet(memory))
    print("Напишите вопрос (Ctrl+C — выход).\n")

    while True:
        try:
            # \ufeff — BOM, который иногда оказывается в начале перенаправленного ввода.
            text = input("Вы > ").strip().lstrip("﻿")
        except (EOFError, KeyboardInterrupt):
            print("\nПока.")
            return
        if not text:
            continue

        if text.startswith("/"):
            if not handle_command(text, memory, state):
                return
            continue

        try:
            print(f"Агент > {run(text, memory, state)}\n")
        except Exception:  # одна неудачная реплика не должна killing весь сеанс
            log.exception("необработанная ошибка на запросе %r", text)
            print("Агент > Произошла ошибка, подробности в agent.log\n")


if __name__ == "__main__":
    # На Windows консоль по умолчанию cp866/cp1251 и не печатает часть символов.
    # Переводим и ввод, и вывод в UTF-8: иначе кириллица в вопросах превращается в "????".
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
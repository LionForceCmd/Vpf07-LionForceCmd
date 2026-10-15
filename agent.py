"""Агент: цикл «Решаю — Делаю — Оцениваю» поверх function calling OpenAI.

Один запуск `run()` — это и есть весь агент:

    1. РЕШАЮ   модель получает системный промпт, память из memory.json и запрос
                пользователя вместе со списком инструментов;
    2. ДЕЛАЮ    если модель вернула вызовы инструментов — агент выполняет их
                и подставляет результаты обратно в диалог;
    3. ОЦЕНИВАЮ модель смотрит на результаты. Если данных хватает — это финальный
                ответ. Если нужны другие инструменты — цикл повторяется.

Цикл ограничен MAX_STEPS: без ограничения модель может бесконечно звать
инструменты по кругу, и агент будет жечь токены, не ответив пользователю.

Побочный эффект шага 3 — весь диалог дописывается в memory.json, поэтому агент
помнит предыдущие запуски. Какие именно инструменты сработали, тоже записываются:
по history видно, на каких данных строился ответ.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from llm import complete, is_offline
from memory import Memory
from tools import TOOL_NAMES, call_tool

log = logging.getLogger("agent")

MAX_STEPS = 8  # максимум итераций цикла на один запрос пользователя

SYSTEM_PROMPT = f"""Ты — локальный AI-агент, который работает в терминале на русском языке.

Твои инструменты: {", ".join(TOOL_NAMES)}.

Правила работы:
1. Данные, которые могут измениться после твоей даты обучения (погода, курсы валют и
   криптовалют, новости, версии библиотек), бери только через инструменты. Не выдумывай их.
2. Не вызывай инструмент, если ответ уже есть в истории диалога. Один вопрос — один-два
   инструмента, не больше.
3. Если данных не хватило или инструмент вернул ошибку — скажи об этом прямо и предложи,
   что можно сделать, вместо того чтобы выдумывать ответ.
4. Инструменты работают с данными, а не с готовым текстом: сначала получи цифры через
   инструмент, потом сам напиши ответ пользователю.
5. Отвечай кратко и по делу. В конце каждого сообщения, где ты дал фактические данные,
   перечисли использованные инструменты в виде: [инструменты: get_weather].
6. Пути к файлам относительные, без указания папки, например notes.txt.
"""


def build_messages(user_text: str, memory: Memory) -> List[Dict[str, Any]]:
    """Системный промпт + память из прошлых запусков + текущий вопрос пользователя."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *memory.history(),
        {"role": "user", "content": user_text},
    ]


def run_step(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Один «Решаю»: спрашиваем у модели, что делать дальше."""
    message = complete(messages)
    messages.append(message)
    log.info(
        "шаг: tool_calls=%s, текст=%d симв.",
        len(message["tool_calls"]),
        len(message["content"]),
    )
    return message


def run_tools(messages: List[Dict[str, Any]]) -> List[str]:
    """Шаг «Делаю»: выполняем все вызовы инструментов и подставляем результаты.

    Ошибка инструмента не прерывает цикл: модель получает текст ошибки и может
    исправить аргументы или выбрать другой инструмент.
    """
    used: List[str] = []

    for call in messages[-1]["tool_calls"]:
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        log.info("инструмент %s, аргументы: %s", name, arguments)

        started = time.monotonic()
        result = call_tool(name, arguments)
        elapsed = time.monotonic() - started

        if result.startswith("ОШИБКА"):
            log.warning("инструмент %s не отработал за %.2f с: %s", name, elapsed, result)
        else:
            log.info("инструмент %s отработал за %.2f с, %d симв.", name, elapsed, len(result))
            used.append(name)

        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    return used


def run(user_text: str, memory: Memory, state: Dict[str, Any] | None = None) -> str:
    """Полный цикл агента для одного запроса. Возвращает текст ответа пользователю.

    state — необязательный словарь; агент кладёт в него ключ "messages" со всем
    диалогом цикла, чтобы CLI мог выгрузить его по команде /transcript.
    """
    messages = build_messages(user_text, memory)
    log.info("запрос: %r (в контексте %d сообщений памяти)", user_text, len(memory.history()))
    memory.add("user", user_text)

    tools_used: List[str] = []
    answer = ""

    for step in range(1, MAX_STEPS + 1):
        message = run_step(messages)
        tools_used.extend(run_tools(messages))

        if not message["tool_calls"]:
            answer = message["content"] or "Инструменты отработали, но пустой ответ."
            break

        log.info("шаг %d: модель запросила %d инструмент(ов)", step, len(message["tool_calls"]))
    else:
        # Лимит шагов: не молчим, а честно говорим, что задача не закрыта.
        answer = (
            f"Не смог завершить задачу за {MAX_STEPS} шагов — инструменты зациклились.\n"
            f"Использовано: {', '.join(tools_used) or 'ничего'}.\n"
            f"Последний результат инструмента: {messages[-1].get('content', '')[:800]}"
        )

    memory.add("assistant", answer, tools_used=sorted(set(tools_used)))
    if state is not None:
        state["messages"] = messages
    log.info("ответ (%d симв.), инструменты: %s", len(answer), tools_used or "нет")
    return answer


def greet(memory: Memory) -> str:
    """Приветствие для CLI: показывает режим работы и состояние памяти."""
    stats = memory.stats()
    mode = (
        "offline-режим: OPENAI_API_KEY не задан, решения принимает локальный планировщик"
        if is_offline()
        else "режим OpenAI: модель выбирает инструменты сама"
    )
    tools = "\n".join(f"  {i + 1}. {name}" for i, name in enumerate(TOOL_NAMES))
    return (
        f"AI-агент готов.\n{mode}\n"
        f"Инструменты:\n{tools}\n"
        f"Память: {stats['file']}\n"
        f"  сессий: {stats['sessions']}, сообщений: {stats['messages']}, "
        f"в контексте: {stats['in_context']}\n"
        "Команды: /tools, /stats, /memory, /clear, /help, /exit"
    )


def dump_transcript(messages: List[Dict[str, Any]]) -> str:
    """Полный диалог цикла в JSON — удобно приложить к багу или показать проверяющему."""
    return json.dumps(messages, ensure_ascii=False, indent=2, default=str)
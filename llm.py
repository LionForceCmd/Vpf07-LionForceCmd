"""Слой обращения к LLM: вызов OpenAI и офлайн-режим без ключа.

Два пути, оба возвращают одинаковые объекты — агенту не важно, отвечает ли модель
или локальный планировщик:

  * с OPENAI_API_KEY — настоящий вызов Chat Completions с function calling;
  * без ключа — offline_plan(): простейший планировщик на регулярных выражениях,
    который выбирает инструмент по тексту запроса и собирает ответ из результата.

Офлайн-режим нужен не только как запасной: он прогоняет весь цикл агента
(выбор инструмента → вызов → подстановка результата) без токенов, поэтому на нём
можно проверить инструменты и память на обычной машине.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from tools import TOOL_SCHEMAS

_client = None


def is_offline() -> bool:
    """True, если ключа OpenAI нет и агент работает на локальном планировщике."""
    return not os.getenv("OPENAI_API_KEY")


def get_client():
    """Ленивая инициализация OpenAI-клиента — создаётся один раз на процесс."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def reset_client() -> None:
    """Сбрасывает клиент (нужно в тестах, когда ключ меняется на лету)."""
    global _client
    _client = None


def _message(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Собирает сообщение OpenAI из ответа. Агент работает только с этим словарём."""
    choice = get_client().chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        messages=messages,
        tools=tools if tools is not None else TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
    ).choices[0].message

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": choice.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (choice.tool_calls or [])
        ],
    }
    return message


def complete(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Один шаг агентного цикла: модель либо зовёт инструмент, либо даёт ответ."""
    if is_offline():
        return offline_plan(messages)
    return _message(messages, TOOL_SCHEMAS)


# ---------------------------------------------------------------------------
# Офлайн-режим: тот же интерфейс, но решение принимает не модель
# ---------------------------------------------------------------------------

_CITY = r"([А-Яа-яЁёA-Za-z][\w\s-]{1,30})"

# Порядок важен: сначала проверяются более специфичные формулировки,
# чтобы «курс доллара» не ушёл в крипту, а «найди» — в калькулятор.
_RULES = [
    (
        "web_search",
        r"^\s*(найди|поищи|загугли|поиск|search)\b",
        lambda m: {"query": m["text"]},
    ),
    (
        "get_weather",
        r"(погод|дожд|градус|тепло|холодно|ветр)",
        lambda m: {"city": m["city"] or "Москва"},
    ),
    (
        "get_crypto_price",
        r"(bitcoin|btc|ethereum|eth\b|litecoin|ltc|solana|sol\b|doge|крипт|монет"
        r"|биткоин|эфир|лайткоин|солана|догикоин)",
        lambda m: {"coin": m["coin"] or "bitcoin", "currency": m["crypto_currency"]},
    ),
    (
        "get_fx_rate",
        r"(курс|доллар|евро|рубл|валют|usd|eur|rub|\$|€|₽)",
        lambda m: {"base": m["fx_base"], "quote": m["fx_quote"]},
    ),
    (
        "file_read",
        # \b обязателен: без него «посчитай» попадает под «читай» из file_read.
        r"\b(прочитай|прочти|открой|содержимое файла|что в файле)\b",
        lambda m: {"path": m["path"] or ""},
    ),
    (
        "file_write",
        r"\b(запиши|сохрани|создай файл|добавь в файл)\b",
        lambda m: {"path": m["path"] or "", "content": m["text"]},
    ),
    (
        "calculate",
        r"(\d\s*[-+*/^]|посчитай|вычисли|как будет|сколько будет|\()",
        lambda m: {"expression": _extract_expression(m["text"])},
    ),
]

_COIN_WORDS = {
    "bitcoin": "bitcoin", "btc": "btc", "биткоин": "btc",
    "ethereum": "ethereum", "eth": "eth", "эфириум": "eth", "эфир": "eth",
    "litecoin": "litecoin", "ltc": "ltc", "лайткоин": "ltc",
    "solana": "solana", "sol": "sol", "солана": "solana",
    "dogecoin": "dogecoin", "doge": "doge", "догикоин": "doge",
}


# Словарь валют в порядке появления в тексте. В офлайн-режиме по нему
# восстанавливается направление курса: «курс доллара к рублю» -> USD -> RUB.
_CURRENCY_WORDS: Dict[str, str] = {
    "usd": "usd", "доллар": "usd", "доллара": "usd", "долларов": "usd", "долларах": "usd",
    "eur": "eur", "евро": "eur",
    "rub": "rub", "руб": "rub", "рубль": "rub", "рубля": "rub", "рублей": "rub",
    "рублях": "rub", "рубчах": "rub",
    "gbp": "gbp", "фунт": "gbp", "фунта": "gbp", "фунтах": "gbp",
    "jpy": "jpy", "иена": "jpy", "йена": "jpy",
    "$": "usd", "€": "eur", "₽": "rub", "£": "gbp",
}

_CURRENCY_PATTERN = re.compile(
    "|".join(re.escape(word) for word in sorted(_CURRENCY_WORDS, key=len, reverse=True)),
    re.IGNORECASE,
)


def _currencies(text: str) -> List[str]:
    """Коды валют в порядке упоминания, без повторов."""
    seen: List[str] = []
    for match in _CURRENCY_PATTERN.finditer(text):
        code = _CURRENCY_WORDS[match.group(0).lower()]
        if code not in seen:
            seen.append(code)
    return seen


def _nominative(name: str) -> str:
    """«в Казани» -> «Казань». Без этого геокодер ищет «Казани» и не находит город."""
    stripped = name.strip()
    for ending, replacement in (("и", "ь"), ("е", "а"), ("ы", "а"), ("у", "а"), ("ом", "")):
        if stripped.endswith(ending) and len(stripped) > len(ending) + 2:
            return stripped[: -len(ending)] + replacement
    return stripped


_MATH_CHARS = set("0123456789+-*/^().,%_ abcdefghijklmnopqrstuvwxyz")


def _extract_expression(text: str) -> str:
    """«посчитай 15% от 3400» -> «(15/100)*3400».

    Знак процента в Python — это остаток от деления, а в русском «15% от 3400»
    значит умножение на 0.15, поэтому проценты переписываются в явное выражение.
    Всё, что не похоже на математику, выбрасывается, так что слова вокруг
    выражения не мешают. Проверку корректности делает сам инструмент calculate.
    """
    text = re.sub(r"(\d[\d.,]*)\s*%\s*от\s*(\d[\d.,]*)", r"(\1/100)*\2", text)
    tail = re.search(
        r"(посчитай|посчитайте|вычисли|как будет|сколько будет)\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    candidate = tail.group(2) if tail else text
    cleaned = "".join(char for char in candidate if char.lower() in _MATH_CHARS).strip()
    return cleaned or candidate.strip()


def _detect(text: str) -> Dict[str, Any]:
    """Достаёт из свободного текста аргументы инструментов регулярными выражениями."""
    lowered = text.lower()
    found: Dict[str, Any] = {"text": text}

    city = re.search(r"(?:в|во|городе|города)\s+" + _CITY, lowered)
    found["city"] = _nominative(city.group(1)).capitalize() if city else None

    coin = next((value for word, value in _COIN_WORDS.items() if word in lowered), None)
    found["coin"] = coin

    codes = _currencies(text)
    # Крипта: валюта котировки — последняя упомянутая, иначе доллар по умолчанию.
    found["crypto_currency"] = codes[-1] if codes else "usd"
    # Обычный курс: первая упомянутая валюта — базовая, вторая — котируемая.
    # Если валюта одна, котируем в рублях: русскоязычному пользователю это полезнее.
    found["fx_base"] = codes[0] if codes else "usd"
    found["fx_quote"] = codes[1] if len(codes) > 1 else (codes[0] if codes else "rub")
    if found["fx_quote"] == found["fx_base"]:
        found["fx_quote"] = "rub"

    path = re.search(r"([\w./-]+\.[a-zа-я]{2,5})", lowered)
    found["path"] = path.group(1) if path else None
    return found


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    return next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )


def _last_tool_result(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Результат последнего выполненного инструмента — из него собирается финальный ответ."""
    for message in reversed(messages):
        if message.get("role") == "tool":
            try:
                return json.loads(message["content"])
            except ValueError:
                return {"raw": message["content"]}
    return None


def _summarize(result: Dict[str, Any]) -> str:
    """Человекочитаемая выжимка из JSON, который вернул инструмент."""
    if "error" in result or str(result.get("raw", "")).startswith("ОШИБКА"):
        return f"Инструмент не отработал: {result.get('error') or result.get('raw')}"

    if "temperature_c" in result:
        parts = [
            f'{result.get("city", "")}: {result.get("temperature_c")}°C, '
            f'{result.get("description", "")}, ветер {result.get("wind_kmh")} км/ч',
            f'влажность {result.get("humidity_pct")}%, ощущается как {result.get("feels_like_c")}°C',
        ]
        forecast = result.get("forecast") or []
        if forecast:
            parts.append("прогноз: " + ", ".join(
                f'{day["date"]} от {day["t_min"]} до {day["t_max"]}°C'
                for day in forecast[:3]
            ))
        return "Погода (источник: %s): %s" % (result.get("provider", "?"), "; ".join(parts))

    if "price" in result:
        return (
            f'{result.get("coin", "?")} = {result["price"]} {result.get("currency", "")} '
            f'(источник: {result.get("source", "?")})'
        )

    if "rate" in result and "base" in result:
        return (
            f'1 {result["base"]} = {result["rate"]} {result["quote"]} '
            f'(источник: {result.get("provider", "?")}, {result.get("updated_at", "")})'
        )

    if "result" in result:
        return f'{result.get("expression", "")} = {result["result"]}'

    if "results" in result:
        lines = [
            f'• {item.get("title", "")} — {item.get("url", "")}\n  {item.get("snippet", "")}'
            for item in result["results"][:5]
        ]
        return "Поиск «%s»: найдено %d\n%s" % (
            result.get("query", ""), result.get("found", 0), "\n".join(lines),
        )

    if "content" in result:
        return f'Файл {result.get("path", "")} ({result.get("chars", 0)} симв.)\n{result["content"]}'

    if "bytes_written" in result:
        return f'Файл {result.get("path", "")} записан: {result["bytes_written"]} байт.'

    return json.dumps(result, ensure_ascii=False, indent=2)[:1200]


def offline_plan(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Планировщик без модели: выбирает инструмент по тексту и печатает его результат."""
    parsed = _last_tool_result(messages)
    if parsed is not None:
        return {
            "role": "assistant",
            "content": _summarize(parsed) + "\n\n[offline-режим: ответ собран из данных инструмента]",
            "tool_calls": [],
        }

    text = _last_user_text(messages)
    found = _detect(text)

    for name, pattern, build_args in _RULES:
        if re.search(pattern, text.lower()):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"offline_{name}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(build_args(found), ensure_ascii=False),
                        },
                    }
                ],
            }

    return {
        "role": "assistant",
        "content": (
            "[offline-режим: OPENAI_API_KEY не задан, поэтому инструмент не выбирался.]\n"
            f"Запрос: {text}\n"
            "В этом режиме я умею выбирать инструмент по ключевым словам: погода, "
            "курс биткоина, курс доллара, поиск, чтение/запись файла, калькулятор."
        ),
        "tool_calls": [],
    }
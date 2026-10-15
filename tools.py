"""Инструменты агента: реализация функций и их JSON-схемы для function calling.

Каждый инструмент — обычная функция Python, которая возвращает строку.
Эта строка целиком уходит модели в сообщении с ролью "tool", поэтому в ней должно быть
ровно то, что нужно для ответа: данные, а не отладочные комментарии.

Реестр из двух словарей:
    TOOLS        — имя -> функция, по ней агент исполняет вызов модели;
    TOOL_SCHEMAS — имя -> схема, по ней модель вообще знает, что инструмент существует.

Все инструменты, которые ходят в интернет, переживают недоступность провайдера:
у погоды и курсов валют есть запасной источник (см. _weather_wttr и _fx_fallback).
"""

from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path
from typing import Any, Callable, Dict, List

import requests

# Агент работает только внутри папки проекта: путь из аргументов модели не может
# выйти за её пределы через "..". Это же снимает проблему из урока с относительными
# путями — путь всегда приводится к абсолютному относительно WORKSPACE.
WORKSPACE = Path(__file__).resolve().parent

HTTP_TIMEOUT = (10, 30)  # (подключение, чтение) — секунды
USER_AGENT = "vpf07-ai-agent/1.0 (homework)"

MAX_READ_BYTES = 2_000_000  # больше 2 МБ не читаем целиком
MAX_CONTENT_CHARS = 100_000  # ограничение на размер текста, который пишет file_write
MAX_SEARCH_RESULTS = 8


class ToolError(Exception):
    """Ошибка инструмента.

    Агент не падает: текст ошибки возвращается модели, и она может попробовать
    другой аргумент или другой инструмент.
    """


def _json(data: Any) -> str:
    """Единый формат ответа инструмента: компактный JSON без ASCII-экранирования."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _http_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """GET + разбор JSON. Сетевые ошибки превращаются в ToolError с понятным текстом."""
    try:
        response = requests.get(
            url,
            params=params,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
    except requests.RequestException as exc:
        raise ToolError(f"Сеть недоступна ({url}): {exc}") from exc

    if response.status_code != 200:
        raise ToolError(f"{url} вернул HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise ToolError(f"{url} вернул не-JSON ответ") from exc


# ---------------------------------------------------------------------------
# Файлы
# ---------------------------------------------------------------------------

def resolve_path(raw: str) -> Path:
    """Приводит путь к абсолютному и не выпускает его за пределы WORKSPACE."""
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    resolved = candidate.resolve()
    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ToolError(
            f"Путь {resolved} находится вне рабочей папки агента {WORKSPACE}. "
            f"Разрешены только файлы внутри проекта."
        )
    return resolved


def file_read(path: str, max_chars: int = 4000) -> str:
    """Читает текстовый файл и возвращает его содержимое."""
    target = resolve_path(path)
    if not target.exists():
        raise ToolError(f"Файл не найден: {target}")

    if target.stat().st_size > MAX_READ_BYTES:
        raise ToolError(
            f"Файл {target.name} больше {MAX_READ_BYTES // 1000} КБ — прочитать целиком нельзя."
        )

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"{target.name} — не текстовый файл (не удалось декодировать UTF-8)") from exc

    limit = max(100, min(int(max_chars), MAX_CONTENT_CHARS))
    truncated = len(text) > limit
    return _json(
        {
            "path": str(target),
            "chars": len(text),
            "truncated": truncated,
            "content": text[:limit] + ("\n...[обрезано]" if truncated else ""),
        }
    )


def file_write(path: str, content: str, append: bool = False) -> str:
    """Создаёт или перезаписывает текстовый файл, при необходимости создавая папки."""
    target = resolve_path(path)
    if len(content) > MAX_CONTENT_CHARS:
        raise ToolError(
            f"Текст {len(content)} символов длиннее лимита {MAX_CONTENT_CHARS}. "
            f"Сократи содержимое и повтори."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    return _json(
        {
            "path": str(target),
            "bytes_written": len(content.encode("utf-8")),
            "mode": "дописано" if append else "перезаписано",
        }
    )


# ---------------------------------------------------------------------------
# Поиск в интернете (DuckDuckGo)
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Ищет информацию в интернете через DuckDuckGo и возвращает срезку результатов."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ToolError("Не установлен пакет ddgs — выполни: pip install ddgs") from exc

    limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    try:
        with DDGS(timeout=20) as client:
            raw = list(client.text(query, max_results=limit))
    except Exception as exc:  # ddgs бросает разные исключения в зависимости от бэкенда
        raise ToolError(f"DuckDuckGo не ответил: {exc}") from exc

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("href") or item.get("url", ""),
            "snippet": (item.get("body", "") or "")[:300],
        }
        for item in raw[:limit]
    ]
    if not results:
        return _json({"query": query, "found": 0, "results": [], "hint": "Ничего не найдено."})
    return _json({"query": query, "found": len(results), "results": results})


# ---------------------------------------------------------------------------
# Погода: Open-Meteo (основной) + wttr.in (запасной)
# ---------------------------------------------------------------------------

# Коды WMO → человекочитаемое описание. Без этой таблицы модель получит число
# вроде 71 и не сможет сказать пользователю, что это "снег".
WEATHER_CODES: Dict[int, str] = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь",
    51: "морось",
    53: "морось",
    55: "сильная морось",
    56: "ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "снежный заряд",
    86: "сильный снежный заряд",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


def geocode_city(city: str) -> Dict[str, Any]:
    """Превращает название города в координаты через geocoding Open-Meteo."""
    data = _http_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": city, "count": 1, "language": "ru", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        raise ToolError(f"Город «{city}» не найден в базе геокодирования.")
    place = results[0]
    return {
        "name": place.get("name", city),
        "country": place.get("country", ""),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": place.get("timezone", "auto"),
    }


def _weather_open_meteo(city: str) -> Dict[str, Any]:
    """Погода через Open-Meteo: координаты из geocode + прогноз по ним."""
    place = geocode_city(city)
    data = _http_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "forecast_days": 3,
        },
    )

    current = data.get("current", {})
    daily = data.get("daily", {})
    forecast = [
        {
            "date": day,
            "t_min": t_min,
            "t_max": t_max,
            "precipitation_mm": precip,
        }
        for day, t_min, t_max, precip in zip(
            daily.get("time", []),
            daily.get("temperature_2m_min", []),
            daily.get("temperature_2m_max", []),
            daily.get("precipitation_sum", []),
        )
    ]
    code = current.get("weather_code")
    return {
        "provider": "open-meteo.com",
        "city": f'{place["name"]}, {place["country"]}'.strip(", "),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "time": current.get("time", ""),
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_kmh": current.get("wind_speed_10m"),
        "description": WEATHER_CODES.get(code, f"код погоды {code}"),
        "forecast": forecast,
    }


def _first(*values: str) -> Any:
    """Первое непустое значение — в ответах wttr.in нужное поле бывает пустой строкой."""
    for value in values:
        if value:
            return value
    return None


def _weather_wttr(city: str) -> Dict[str, Any]:
    """Запасной источник погоды: wttr.in сам умеет искать город по названию."""
    slug = city.strip().replace(" ", "+")
    data = _http_json(f"https://wttr.in/{slug}", {"format": "j1", "lang": "ru"})
    current = data["current_condition"][0]
    area = data.get("nearest_area", [{}])[0]

    forecast = []
    for day in data.get("weather", [])[:3]:
        hourly = day.get("hourly", [])
        mid = hourly[len(hourly) // 2] if hourly else {}
        forecast.append(
            {
                "date": day.get("date", ""),
                "t_min": day.get("mintempC", ""),
                "t_max": day.get("maxtempC", ""),
                "precipitation_mm": mid.get("precipMM", ""),
            }
        )

    # Русский текст wttr.in кладёт в отдельное поле lang_ru; если сервер его не
    # прислал, в weatherDesc лежит английский вариант того же описания.
    lang_ru = current.get("lang_ru") or []
    description = _first(
        lang_ru[0].get("value", "") if lang_ru else "",
        _first(*[entry.get("value", "") for entry in current.get("weatherDesc", [])]),
        "нет данных",
    )

    return {
        "provider": "wttr.in (запасной)",
        "city": area.get("areaName", [{}])[0].get("value", city),
        "country": area.get("country", [{}])[0].get("value", ""),
        "latitude": area.get("latitude", ""),
        "longitude": area.get("longitude", ""),
        "time": current.get("localObsDateTime", ""),
        "temperature_c": _first(current.get("temp_C", ""), current.get("FeelsLikeC", "")),
        "feels_like_c": _first(current.get("FeelsLikeC", ""), current.get("temp_C", "")),
        "humidity_pct": current.get("humidity", ""),
        "wind_kmh": current.get("windspeedKmph", ""),
        "description": description,
        "forecast": forecast,
    }


def get_weather(city: str) -> str:
    """Погода для любого города: сначала Open-Meteo, при сбое — wttr.in."""
    if not city.strip():
        raise ToolError("Не указано название города.")
    try:
        return _json(_weather_open_meteo(city))
    except (ToolError, KeyError, IndexError) as primary_error:
        try:
            payload = _weather_wttr(city)
        except (ToolError, KeyError, IndexError) as fallback_error:
            raise ToolError(
                f"Не удалось получить погоду для «{city}». "
                f"Open-Meteo: {primary_error}. Запасной источник: {fallback_error}."
            ) from fallback_error
        payload["fallback_reason"] = str(primary_error)
        return _json(payload)


# ---------------------------------------------------------------------------
# Курс криптовалют (CoinGecko, без ключа)
# ---------------------------------------------------------------------------

# Пользователь пишет "btc", а CoinGecko ждёт "bitcoin". Без синонимов модель
# пришлось бы угадывать идентификаторы — здесь они собраны явно.
COIN_ALIASES: Dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "ltc": "litecoin",
    "bnb": "binancecoin",
    "sol": "solana",
    "xrp": "ripple",
    "doge": "dogecoin",
    "ada": "cardano",
    "dot": "polkadot",
    "trx": "tron",
    "usdt": "tether",
    "ton": "the-open-network",
    "monero": "monero",
    "zcash": "zcash",
}

CURRENCY_ALIASES: Dict[str, str] = {
    "$": "usd",
    "€": "eur",
    "£": "gbp",
    "₽": "rub",
    "руб": "rub",
    "доллар": "usd",
    "евро": "eur",
    "фунт": "gbp",
    "доллар сша": "usd",
}


def _normalize_currency(currency: str) -> str:
    value = (currency or "usd").strip().lower()
    return CURRENCY_ALIASES.get(value, value)


def get_crypto_price(coin: str, currency: str = "usd") -> str:
    """Текущая цена криптовалюты в обычной валюте."""
    if not coin.strip():
        raise ToolError("Не указана монета.")

    coin_id = COIN_ALIASES.get(coin.strip().lower(), coin.strip().lower())
    currency_id = _normalize_currency(currency)

    data = _http_json(
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids": coin_id, "vs_currencies": currency_id, "include_last_updated_at": "true"},
    )
    if coin_id not in data:
        raise ToolError(
            f"Монета «{coin}» не найдена в CoinGecko. Известные сокращения: "
            f"{', '.join(sorted(COIN_ALIASES))}. Попробуй полное название латиницей."
        )

    quote = data[coin_id].get(currency_id)
    if quote is None:
        raise ToolError(
            f"Курс {coin_id} к {currency_id} недоступен. Попробуй usd, eur или rub."
        )

    updated = data[coin_id].get("last_updated_at")
    return _json(
        {
            "coin": coin_id,
            "currency": currency_id.upper(),
            "price": quote,
            "last_updated_at": updated,
            "source": "CoinGecko",
        }
    )


# ---------------------------------------------------------------------------
# Курс обычных валют: open.er-api.com (основной) + Frankfurter/ECB (запасной)
# ---------------------------------------------------------------------------

# Frankfurter — это данные ЕЦБ, а ЦБ ЕС с 2022 года не публикует курс рубля,
# поэтому рубль там отсутствует. Именно поэтому основной источник другой.
FX_FALLBACK_CURRENCIES = {"eur", "usd", "gbp", "jpy", "chf", "cny", "try", "pln"}


def _fx_primary(base: str, quote: str) -> Dict[str, Any]:
    data = _http_json(f"https://open.er-api.com/v6/latest/{base}")
    if data.get("result") != "success":
        raise ToolError(f"Курсы не получены: {data.get('error-type', 'неизвестная ошибка')}")
    rates = data.get("rates", {})
    # Курсы приходят ключами в верхнем регистре ("RUB"), а не как запросили.
    if quote not in rates:
        raise ToolError(f"Валюта {quote} недоступна в источнике курсов.")
    return {
        "provider": "open.er-api.com",
        "base": base.upper(),
        "quote": quote.upper(),
        "rate": rates[quote],
        "updated_at": data.get("time_last_update_utc", ""),
    }


def _fx_fallback(base: str, quote: str) -> Dict[str, Any]:
    if base not in FX_FALLBACK_CURRENCIES or quote not in FX_FALLBACK_CURRENCIES:
        raise ToolError(
            "Рубля нет в данных ЕЦБ (Frankfurter) — этот запасной источник "
            "поддерживает только " + ", ".join(sorted(FX_FALLBACK_CURRENCIES)) + "."
        )
    data = _http_json("https://api.frankfurter.app/latest", {"from": base, "to": quote})
    rates = data.get("rates", {})
    if quote not in rates:
        raise ToolError(f"Курс {base}→{quote} недоступен в ЕЦБ.")
    return {
        "provider": "Frankfurter / ЕЦБ (запасной)",
        "base": base,
        "quote": quote,
        "rate": rates[quote],
        "updated_at": data.get("date", ""),
    }


def get_fx_rate(base: str = "usd", quote: str = "rub") -> str:
    """Курс обычной валюты, например EUR→RUB или USD→RUB."""
    base_code = (base or "usd").strip().lower()
    quote_code = _normalize_currency(quote)

    if not base_code.isalpha() or len(base_code) != 3:
        raise ToolError("Базовая валюта — трёхбуквенный код, например USD или EUR.")
    if not quote_code.isalpha() or len(quote_code) != 3:
        raise ToolError("Валюта котировки — трёхбуквенный код, например RUB или EUR.")

    # Запасной источник проверяется по строчным кодам, а запросы уходят в верхнем
    # регистре — именно в таком виде курсы лежат в ответах обоих API.
    try:
        return _json(_fx_primary(base_code.upper(), quote_code.upper()))
    except (ToolError, KeyError) as primary_error:
        try:
            payload = _fx_fallback(base_code.upper(), quote_code.upper())
        except (ToolError, KeyError) as fallback_error:
            raise ToolError(
                f"Не удалось получить курс {base_code.upper()}→{quote_code.upper()}. "
                f"Основной источник: {primary_error}. Запасной: {fallback_error}."
            ) from fallback_error
        payload["fallback_reason"] = str(primary_error)
        return _json(payload)


# ---------------------------------------------------------------------------
# Калькулятор
# ---------------------------------------------------------------------------

_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Только математика: никаких импортов, атрибутов и вызовов вроде open().
_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "log": math.log10,
    "ln": math.log,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "hypot": math.hypot,
}
_CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError(f"Недопустимая константа: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError(f"Недопустимая функция: {ast.dump(node.func)[:60]}")
        if node.keywords:
            raise ToolError("Именованные аргументы не поддерживаются.")
        return _FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ToolError(
        f"Недопустимое выражение: {type(node).__name__}. "
        f"Разрешены числа, + - * / // % **, скобки, функции {', '.join(sorted(_FUNCTIONS))}."
    )


def calculate(expression: str) -> str:
    """Считает арифметическое выражение. Код выполнения не запускается — только арифметика."""
    text = expression.strip()
    if not text:
        raise ToolError("Пустое выражение.")
    if len(text) > 500:
        raise ToolError("Выражение длиннее 500 символов.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Не удалось разобрать выражение: {exc.msg}") from exc

    try:
        value = _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Деление на ноль.") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"Выражение не вычислилось: {exc}") from exc

    return _json({"expression": text, "result": value})


# ---------------------------------------------------------------------------
# Реестр инструментов
# ---------------------------------------------------------------------------

TOOLS: Dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "get_weather": get_weather,
    "get_crypto_price": get_crypto_price,
    "file_read": file_read,
    "file_write": file_write,
    "get_fx_rate": get_fx_rate,
    "calculate": calculate,
}


def _schema(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: List[str],
) -> Dict[str, Any]:
    """Собирает одну схему инструмента в формате, который ждёт OpenAI."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _schema(
        "web_search",
        "Поиск актуальной информации в интернете через DuckDuckGo. "
        "Используй для любых вопросов о фактах, которые могли измениться после обучения модели. "
        "Возвращает заголовки, ссылки и краткие выдержки из найденных страниц.",
        {
            "query": {
                "type": "string",
                "description": "Поисковый запрос на языке пользователя.",
            },
            "max_results": {
                "type": "integer",
                "description": "Сколько результатов вернуть, 1-8. По умолчанию 5.",
            },
        },
        ["query"],
    ),
    _schema(
        "get_weather",
        "Текущая погода и прогноз на 3 дня для любого города по его названию.",
        {
            "city": {
                "type": "string",
                "description": "Название города: Москва, Казань, Санкт-Петербург.",
            }
        },
        ["city"],
    ),
    _schema(
        "get_crypto_price",
        "Текущий курс криптовалюты в обычной валюте.",
        {
            "coin": {
                "type": "string",
                "description": "Монета: bitcoin, ethereum, btc, eth, solana, dogecoin.",
            },
            "currency": {
                "type": "string",
                "description": "Валюта котировки: usd, eur, rub. По умолчанию usd.",
            },
        },
        ["coin"],
    ),
    _schema(
        "get_fx_rate",
        "Курс обычной валюты по данным ЦБ/ЕЦБ, например EUR→RUB или USD→RUB.",
        {
            "base": {
                "type": "string",
                "description": "Что продаём: usd, eur, gbp. По умолчанию usd.",
            },
            "quote": {
                "type": "string",
                "description": "В что переводим: rub, usd, eur. По умолчанию rub.",
            },
        },
        [],
    ),
    _schema(
        "file_read",
        "Читает текстовый файл из рабочей папки агента и возвращает его содержимое.",
        {
            "path": {
                "type": "string",
                "description": "Путь к файлу относительно папки проекта, например notes.txt.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Сколько символов вернуть, по умолчанию 4000.",
            },
        },
        ["path"],
    ),
    _schema(
        "file_write",
        "Записывает текст в файл рабочей папки агента (создаёт файл и папки при необходимости).",
        {
            "path": {
                "type": "string",
                "description": "Путь к файлу относительно папки проекта, например report.md.",
            },
            "content": {
                "type": "string",
                "description": "Текст, который нужно записать в файл.",
            },
            "append": {
                "type": "boolean",
                "description": "True — дописать в конец файла, False — перезаписать (по умолчанию).",
            },
        },
        ["path", "content"],
    ),
    _schema(
        "calculate",
        "Точный калькулятор для арифметики, степеней и математических функций.",
        {
            "expression": {
                "type": "string",
                "description": "Выражение: (2+3)*sqrt(16), или sin(pi/2), или 10**3.",
            }
        },
        ["expression"],
    ),
]


def tool_names() -> List[str]:
    """Список имён инструментов — для логов и системного промпта."""
    return list(TOOLS)


def call_tool(name: str, arguments: str) -> str:
    """Исполняет вызов модели. Никогда не бросает — модель получает текст ошибки."""
    function = TOOLS.get(name)
    if function is None:
        return f"ОШИБКА: нет инструмента «{name}». Доступны: {', '.join(TOOL_NAMES)}."

    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
    except ValueError as exc:
        return f"ОШИБКА: аргументы инструмента «{name}» — не корректный JSON ({exc})."
    if not isinstance(parsed, dict):
        return f"ОШИБКА: аргументы инструмента «{name}» должны быть JSON-объектом."

    try:
        return function(**parsed)
    except ToolError as exc:
        return f"ОШИБКА ({name}): {exc}"
    except TypeError as exc:
        return f"ОШИБКА ({name}): неверные аргументы ({exc})."
    except Exception as exc:  # сеть/библиотеки могут бросить что угодно — агент не должен падать
        return f"ОШИБКА ({name}): {type(exc).__name__}: {exc}"


TOOL_NAMES = tool_names()
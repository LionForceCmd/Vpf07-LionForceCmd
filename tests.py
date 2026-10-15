"""Проверки агента без сети и без OPENAI_API_KEY.

Запуск:
    python tests.py

Что проверяется:
    * инструменты, схемы function calling и защита файловой системы;
    * калькулятор, включая попытку выполнить чужой код;
    * долговременная память: перезапись файла, окно контекста, восстановление;
    * офлайн-планировщик: выбор инструмента по тексту запроса;
    * агентный цикл: сборка запроса, исполнение инструментов, лимит шагов.

Сеть не используется: инструменты, ходящие в интернет, подменяются заглушками.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import agent as agent_module
import llm
import tools
from memory import Memory

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool) -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok — {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL — {name}")


def raises(exception: type[Exception], call, *args, **kwargs) -> bool:
    """True, если вызов упал именно ожидаемым исключением."""
    try:
        call(*args, **kwargs)
    except exception:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# 1. Реестр инструментов и схемы function calling
# ---------------------------------------------------------------------------

def test_registry() -> None:
    print("\n[1] Реестр инструментов")
    schema_names = [s["function"]["name"] for s in tools.TOOL_SCHEMAS]

    check("схемы уникальны", len(schema_names) == len(set(schema_names)))
    check("в TOOLS и TOOL_SCHEMAS одно и то же множество", set(tools.TOOLS) == set(schema_names))
    check("все 7 инструментов из задания на месте", len(tools.TOOLS) == 7)

    valid = True
    for schema in tools.TOOL_SCHEMAS:
        function = schema["function"]
        parameters = function["parameters"]
        valid &= schema["type"] == "function"
        valid &= parameters["type"] == "object"
        valid &= bool(function["description"].strip())
        valid &= set(parameters["required"]) <= set(parameters["properties"])
    check("схемы в формате OpenAI (type/description/required ⊆ properties)", valid)

    check("все инструменты возвращают строку", all(
        isinstance(result, str)
        for result in [
            tools.calculate("1+1"),
            tools.call_tool("get_crypto_price", '{"coin": "btc"}'),
            tools.call_tool("get_weather", '{"city": "Москва"}'),
            tools.call_tool("web_search", '{"query": "test"}'),
            tools.call_tool("get_fx_rate", '{"base": "usd", "quote": "rub"}'),
        ]
    ))


# ---------------------------------------------------------------------------
# 2. Калькулятор
# ---------------------------------------------------------------------------

def test_calculator() -> None:
    print("\n[2] Калькулятор")
    check("арифметика и скобки", json.loads(tools.calculate("(2+3)*sqrt(16)"))["result"] == 20.0)
    check("степень", json.loads(tools.calculate("2**10"))["result"] == 1024.0)
    check("приоритет операций", json.loads(tools.calculate("2+3*4"))["result"] == 14.0)
    check("константы pi", abs(json.loads(tools.calculate("pi"))["result"] - 3.14159) < 0.001)
    check("деление на ноль -> ToolError", raises(tools.ToolError, tools.calculate, "1/0"))
    check("мусор -> ToolError", raises(tools.ToolError, tools.calculate, "сколько будет"))

    # Ключевое: калькулятор не должен выполнять произвольный код.
    dangerous = [
        "__import__('os').system('echo hacked')",
        "open('secret.txt').read()",
        "(1).__class__",
        "[x for x in range(10)]",
    ]
    blocked = all(raises(tools.ToolError, tools.calculate, expression) for expression in dangerous)
    check("выполнение произвольного кода заблокировано", blocked)


# ---------------------------------------------------------------------------
# 3. Файлы
# ---------------------------------------------------------------------------

def test_files() -> None:
    print("\n[3] Файлы и защита рабочей папки")

    with tempfile.TemporaryDirectory(dir=str(tools.WORKSPACE)) as temp:
        base = Path(temp)

        relative = tools.resolve_path("notes.txt")
        check("относительный путь становится абсолютным", relative == tools.WORKSPACE / "notes.txt")
        check("путь внутри WORKSPACE принимается", tools.resolve_path(str(base / "a.txt")) == base / "a.txt")

        escapes = ["../outside.txt", "../../../Windows/win.ini", "C:\\Windows\\win.ini"]
        blocked = all(raises(tools.ToolError, tools.resolve_path, path) for path in escapes)
        check("выход за пределы проекта заблокирован", blocked)

        target = str(base / "заметка.txt")
        tools.file_write(target, "привет, агент")
        check("запись файла", (base / "заметка.txt").read_text(encoding="utf-8") == "привет, агент")

        tools.file_write(target, " и продолжение", append=True)
        check(
            "дописание в файл",
            (base / "заметка.txt").read_text(encoding="utf-8") == "привет, агент и продолжение",
        )

        payload = json.loads(tools.file_read(target))
        check("чтение файла возвращает содержимое", payload["content"] == "привет, агент и продолжение")
        check("в ответе указан абсолютный путь", Path(payload["path"]).is_absolute())

        tools.file_write(str(base / "длинный.txt"), "я" * 500)
        limited = json.loads(tools.file_read(str(base / "длинный.txt"), max_chars=100))
        check("лимит символов при чтении", limited["truncated"] and len(limited["content"]) < 200)

        check("чтение несуществующего файла -> ToolError", raises(
            tools.ToolError, tools.file_read, str(base / "нет.txt")))

        huge = "x" * (tools.MAX_CONTENT_CHARS + 1)
        check("слишком большой текст отклоняется", raises(
            tools.ToolError, tools.file_write, str(base / "big.txt"), huge))

        nested = str(base / "вложенная" / "папка" / "файл.txt")
        tools.file_write(nested, "ok")
        check("создание промежуточных папок", Path(nested).read_text(encoding="utf-8") == "ok")


# ---------------------------------------------------------------------------
# 4. Нормализация названий
# ---------------------------------------------------------------------------

def test_normalization() -> None:
    print("\n[4] Нормализация названий монет и валют")
    check("btc -> bitcoin", tools.COIN_ALIASES["btc"] == "bitcoin")
    check("ltc -> litecoin", tools.COIN_ALIASES["ltc"] == "litecoin")
    check("символ ₽ -> rub", tools._normalize_currency("₽") == "rub")
    check("слово «доллар» -> usd", tools._normalize_currency("доллар") == "usd")
    check("код не меняется", tools._normalize_currency("EUR") == "eur")
    check("код RUB -> rub", tools._normalize_currency("RUB") == "rub")


# ---------------------------------------------------------------------------
# 5. Долговременная память
# ---------------------------------------------------------------------------

def test_memory() -> None:
    print("\n[5] Память в memory.json")

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "memory.json"

        mem = Memory(path=path)
        check("на диске памяти ещё нет", not path.exists())
        check("создаётся первая сессия", mem.session["id"] == 1)

        mem.add("user", "Меня зовут Аня")
        mem.add("assistant", "Привет, Аня!", tools_used=["web_search"])
        check("файл памяти создан", path.exists())

        stored = json.loads(path.read_text(encoding="utf-8"))
        check("формат файла: version/sessions", stored["version"] == 1 and "sessions" in stored)
        check("сообщение сохранено", stored["sessions"][0]["messages"][0]["content"] == "Меня зовут Аня")
        check(
            "использованные инструменты записаны",
            stored["sessions"][0]["messages"][1]["tools_used"] == ["web_search"],
        )

        restored = Memory(path=path)
        check("история переживает перезапуск", restored.session["messages"][0]["content"] == "Меня зовут Аня")
        check("в контекст попадает прошлая реплика", restored.history()[0]["content"] == "Меня зовут Аня")

        mem.add("user", "   ")
        check("пустое сообщение не сохраняется", len(mem.session["messages"]) == 2)

        for index in range(30):
            mem.add("user", f"сообщение {index}")
        check("окно контекста ограничено", len(mem.history()) == mem.context_messages)
        check("в окне лежат последние сообщения", mem.history()[-1]["content"] == "сообщение 29")

        # Сессия продолжается только в день своего начала.
        yesterday = (datetime.now() - timedelta(days=1)).isoformat(timespec="seconds")
        stored["sessions"][-1]["started_at"] = yesterday
        path.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
        resumed = Memory(path=path)
        check("на другой день начинается новая сессия", resumed.session["id"] == 2)

        mem.clear()
        check("очистка памяти", mem.stats()["messages"] == 0)
        check("файл после очистки валиден", json.loads(path.read_text(encoding="utf-8"))["version"] == 1)

        path.write_text("{ это не json", encoding="utf-8")
        recovered = Memory(path=path)
        check("битый memory.json не роняет агента", recovered.stats()["messages"] == 0)
        check("битый файл отложен для разбора", path.with_suffix(".json.broken").exists())


# ---------------------------------------------------------------------------
# 6. Офлайн-планировщик выбора инструментов
# ---------------------------------------------------------------------------

def test_offline_planner() -> None:
    print("\n[6] Офлайн-планировщик")

    def plan(text: str) -> tuple[str | None, dict]:
        message = llm.offline_plan([{"role": "user", "content": text}])
        calls = message["tool_calls"]
        if not calls:
            return None, {}
        arguments = json.loads(calls[0]["function"]["arguments"])
        return calls[0]["function"]["name"], arguments

    name, args = plan("какая погода в Казани")
    check("погода -> get_weather", name == "get_weather" and args["city"] == "Казань")

    name, args = plan("какой курс доллара к рублю")
    check("курс -> get_fx_rate", name == "get_fx_rate")
    check("направление курса USD->RUB", args == {"base": "usd", "quote": "rub"})

    name, args = plan("курс евро к доллару")
    check("курс EUR->USD", args == {"base": "eur", "quote": "usd"})

    name, args = plan("сколько стоит 1 bitcoin")
    check("крипта -> get_crypto_price", name == "get_crypto_price" and args["coin"] == "bitcoin")

    name, args = plan("а сколько litecoin в rub")
    check("крипта с валютой из запроса", args["currency"] == "rub")

    name, args = plan("найди информацию про Эрмитаж")
    check("поиск -> web_search", name == "web_search")

    name, args = plan("прочитай notes.txt")
    check("чтение файла", name == "file_read" and args["path"] == "notes.txt")

    # Регрессия: «посчитай» содержит подстроку «читай».
    name, args = plan("посчитай (2+3)*sqrt(16)")
    check("«посчитай» не уходит в file_read", name == "calculate")
    check("выражение вытащено целиком", args["expression"] == "(2+3)*sqrt(16)")

    # После результата инструмента планировщик отвечает текстом, а не крутит цикл.
    after_tool = llm.offline_plan([
        {"role": "user", "content": "посчитай 2+2"},
        {"role": "tool", "content": json.dumps({"expression": "2+2", "result": 4})},
    ])
    check("после инструмента — текстовый ответ", not after_tool["tool_calls"])
    check("ответ содержит результат", "4" in after_tool["content"])


# ---------------------------------------------------------------------------
# 7. Агентный цикл
# ---------------------------------------------------------------------------

FAKE_GEOCODING = {
    "results": [{
        "name": "Москва",
        "country": "Россия",
        "latitude": 55.752,
        "longitude": 37.616,
        "timezone": "Europe/Moscow",
    }]
}

FAKE_FORECAST = {
    "current": {
        "time": "2026-10-15T12:00",
        "temperature_2m": 7,
        "apparent_temperature": 5,
        "relative_humidity_2m": 80,
        "weather_code": 61,  # по таблице WMO это «небольшой дождь»
        "wind_speed_10m": 8,
    },
    "daily": {
        "time": ["2026-10-15"],
        "temperature_2m_min": [4],
        "temperature_2m_max": [9],
        "precipitation_sum": [1.2],
    },
}


def fake_http(url: str, params: dict | None = None) -> dict:
    """Заглушка Open-Meteo: разводит геокодер и прогноз по URL.

    Благодаря разветвлению тест проверяет настоящий код geocode_city и
    _weather_open_meteo, включая перевод кода погоды в текст.
    """
    if "geocoding" in url:
        return FAKE_GEOCODING
    if "forecast" in url:
        return FAKE_FORECAST
    raise tools.ToolError(f"тест не ожидал запрос к {url}")


def test_agent_loop() -> None:
    print("\n[7] Агентный цикл")

    with tempfile.TemporaryDirectory() as temp:
        memory = Memory(path=Path(temp) / "memory.json")

        messages = agent_module.build_messages("привет", memory)
        check("system-промпт первый", messages[0]["role"] == "system")
        check("запрос пользователя последний", messages[-1]["content"] == "привет")
        check("в промпте перечислены инструменты", "get_weather" in messages[0]["content"])

        memory.add("user", "Меня зовут Аня")
        memory.add("assistant", "Привет, Аня!")
        with_history = agent_module.build_messages("как меня зовут?", memory)
        check("память подставляется в запрос", "Меня зовут Аня" in with_history[1]["content"])

    # Полный проход цикла с подменённой сетью: настоящий агент, фальшивые данные.
    original_http = tools._http_json
    tools._http_json = fake_http
    try:
        with tempfile.TemporaryDirectory() as temp:
            memory = Memory(path=Path(temp) / "memory.json")
            state: dict = {}

            answer = agent_module.run("какая погода в Москве", memory, state)

            check("агент вернул текстовый ответ", isinstance(answer, str) and bool(answer.strip()))
            check("ответ содержит температуру из инструмента", "7" in answer)
            check("код погоды переведён в текст", "дождь" in answer)
            check("в памяти записаны оба сообщения", memory.stats()["messages"] == 2)
            check(
                "в памяти указан использованный инструмент",
                memory.session["messages"][-1].get("tools_used") == ["get_weather"],
            )
            check(
                "каждое сообщение роли tool связано с вызовом",
                all(message.get("tool_call_id") for message in state["messages"] if message["role"] == "tool"),
            )
            check("в запросе не осталось сообщений без ответа", not any(
                message.get("tool_calls") for message in state["messages"][-2:]
            ))
    finally:
        tools._http_json = original_http

    # Ошибка инструмента не должна обрывать цикл: модель читает текст ошибки.
    tool_calls = {"count": 0}
    model_steps = {"count": 0}

    def always_failing_call(name: str, arguments: str) -> str:
        tool_calls["count"] += 1
        return f"ОШИБКА ({name}): сервис недоступен"

    original_call_tool = agent_module.call_tool
    original_complete = agent_module.complete

    def looping_complete(messages):
        """Модель, которая упрямо зовёт инструмент и никогда не отвечает сама."""
        model_steps["count"] += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{model_steps['count']}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Москва"}'},
            }],
        }

    agent_module.call_tool = always_failing_call
    agent_module.complete = looping_complete
    try:
        with tempfile.TemporaryDirectory() as temp:
            memory = Memory(path=Path(temp) / "memory.json")
            answer = agent_module.run("зацикли меня", memory)

            check("цикл останавливается на MAX_STEPS", model_steps["count"] == agent_module.MAX_STEPS)
            check("инструмент вызван на каждом шаге", tool_calls["count"] == agent_module.MAX_STEPS)
            check("агент честно сообщает о лимите шагов", "шагов" in answer)
            check("после лимита ответ всё равно записан в память", memory.stats()["messages"] == 2)
    finally:
        agent_module.call_tool = original_call_tool
        agent_module.complete = original_complete


class FakeOpenAI:
    """Заглушка SDK: повторяет структуру ответа Chat Completions с вызовом инструмента."""

    last_request: dict = {}

    class _Function:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class _Call:
        def __init__(self, name: str, arguments: str) -> None:
            self.id = "call_openai_1"
            self.function = FakeOpenAI._Function(name, arguments)

    class _Message:
        def __init__(self) -> None:
            self.content = ""
            self.tool_calls = [FakeOpenAI._Call("get_weather", '{"city": "Москва"}')]

    class _Choice:
        def __init__(self) -> None:
            self.message = FakeOpenAI._Message()

    class _Completion:
        def __init__(self) -> None:
            self.choices = [FakeOpenAI._Choice()]

    class _Completions:
        def create(self, **kwargs):
            FakeOpenAI.last_request = kwargs
            return FakeOpenAI._Completion()

    def __init__(self) -> None:
        self.chat = type("_Chat", (), {"completions": FakeOpenAI._Completions()})()


def test_openai_path() -> None:
    """Проверяет сборку запроса к OpenAI: без него весь основной путь не покрыт тестами."""
    print("\n[8] Запрос к OpenAI")

    fake = FakeOpenAI()
    original_client = llm.get_client
    original_offline = llm.is_offline
    original_env = os.environ.get("OPENAI_API_KEY")

    llm.get_client = lambda: fake
    llm.is_offline = lambda: False
    os.environ["OPENAI_API_KEY"] = "sk-тест"
    try:
        message = llm.complete([{"role": "user", "content": "погода в Москве"}])

        request = fake.last_request
        check("модель и temperature заданы", "model" in request and "temperature" in request)
        check("схемы инструментов переданы в запрос", request["tools"] == tools.TOOL_SCHEMAS)
        check("выбор инструмента делегирован модели", request["tool_choice"] == "auto")
        check("сообщения пользователя дошли до запроса", request["messages"][0]["content"] == "погода в Москве")
        check("ответ разобран в tool_calls", message["tool_calls"][0]["function"]["name"] == "get_weather")
        check("аргументы вызова сохранены", message["tool_calls"][0]["function"]["arguments"] == '{"city": "Москва"}')
        check("id вызова сохранён для ответа tool", message["tool_calls"][0]["id"] == "call_openai_1")
        check("у вызова есть тип function", message["tool_calls"][0]["type"] == "function")
    finally:
        llm.get_client = original_client
        llm.is_offline = original_offline
        if original_env is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original_env


def main() -> None:
    print("Проверки AI-агента (сеть и OpenAI не используются)\n" + "=" * 55)
    # Агент логирует каждый шаг; при прогоне тестов эти сообщения только мешают.
    logging.disable(logging.CRITICAL)

    test_registry()
    test_calculator()
    test_files()
    test_normalization()
    test_memory()
    test_offline_planner()
    test_agent_loop()
    test_openai_path()

    print("\n" + "=" * 55)
    if FAILED:
        print(f"Провалено: {len(FAILED)} из {PASSED + len(FAILED)}")
        for name in FAILED:
            print(f"  - {name}")
        raise SystemExit(1)
    print(f"Все проверки пройдены: {PASSED}")


if __name__ == "__main__":
    main()
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from yaml import CDumper

from src.common.config.config import settings
from src.infrastructure.logger import get_logger

from .client import langfuse

logger = get_logger(__name__)

PROJECT_NAME = settings.get("PROJECT_NAME", "default")
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "agents" / "prompts"
PROMPTS_DIR = Path(settings.get("PROMPTS_DIR") or _DEFAULT_PROMPTS_DIR)
LANGFUSE_ENABLED = settings.get("LANGFUSE_ENABLED", False)


# Паттерн для поиска переменных в формате {{variable_name}}
_VAR_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def _str_presenter(dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter)
yaml.add_representer(str, _str_presenter, Dumper=yaml.SafeDumper)
yaml.add_representer(str, _str_presenter, Dumper=CDumper)


def qualify_prompt_name(prompt_name, project_name=None):
    """
    Формирует полное имя промпта с указанием проекта.

    Если имя уже содержит "/", возвращает его без изменений.
    Иначе добавляет имя проекта в начало.

    Args:
        prompt_name: Имя промпта (может быть с проектом или без)
        project_name: Имя проекта (если None, используется из env)

    Returns:
        Полное имя промпта в формате "project/name"
    """
    if "/" in prompt_name:
        return prompt_name
    project = project_name or PROJECT_NAME
    return f"{project}/{prompt_name}"


@dataclass
class Prompt:
    """
    Класс для представления промпта.

    Поддерживает два типа промптов:
    - text: простой текстовый промпт (строка)
    - chat: промпт для чата (список сообщений)

    Поддерживает три формата представления:
    - string: строка (только для text)
    - dict_list: список словарей [{"role": "user", "content": "..."}]
    - tuple_list: список кортежей [("user", "..."), ("assistant", "...")]

    Attributes:
        name: Имя промпта
        project: Имя проекта
        type: Тип промпта ("text" или "chat")
        prompt: Содержимое промпта (строка или список)
        prompt_format: Формат промпта ("string", "dict_list" или "tuple_list")
        config: Дополнительная конфигурация
        version: Версия промпта
        labels: Список меток
        tags: Список тегов
    """

    name: str | None = None
    project: str | None = None
    type: str = "text"
    prompt: str | list = ""
    prompt_format: str | None = None
    config: dict | None = None
    version: int | None = None
    labels: list = field(default_factory=lambda: ["production"])
    tags: list = field(default_factory=list)
    lf_raw: dict | None = None

    def __post_init__(self):
        """
        Инициализация после создания объекта.

        Выполняет нормализацию имени проекта и автоматическое определение
        типа и формата промпта на основе его содержимого.
        """
        # Нормализация имени проекта: извлекаем проект из имени, если оно содержит "/"
        if self.name and self.project is None and "/" in self.name:
            project, short_name = self.name.split("/", 1)
            self.project = project
            self.name = short_name
        # Если проект не указан, используем проект по умолчанию из env
        if self.project is None:
            self.project = PROJECT_NAME
        # Определение типа и формата промпта
        if self.prompt_format is None:
            # Автоматическое определение на основе содержимого
            self._detect_and_set_format()
        # Если формат указан явно, определяем тип и конвертируем при необходимости
        elif isinstance(self.prompt, str):
            self.type = "text"
            if self.prompt_format != "string":
                raise ValueError(f"Для text промпта формат должен быть 'string', получен '{self.prompt_format}'")
        elif isinstance(self.prompt, list):
            self.type = "chat"
            # Если указан dict_list, конвертируем в tuple_list
            if self.prompt_format == "dict_list":
                if self.prompt:
                    self.prompt = [(msg.get("role", "user"), msg.get("content", "")) for msg in self.prompt]
                self.prompt_format = "tuple_list"
            elif self.prompt_format != "tuple_list":
                raise ValueError(f"Для chat промпта формат должен быть 'tuple_list', получен '{self.prompt_format}'")
        else:
            raise ValueError(f"Неподдерживаемый тип данных промпта: {type(self.prompt)}")

    def _detect_and_set_format(self):
        """
        Автоматически определяет тип и формат промпта на основе его содержимого.

        Анализирует структуру данных промпта и устанавливает:
        - type: "text" для строки, "chat" для списка
        - prompt_format: "string", "dict_list" или "tuple_list"

        Raises:
            ValueError: Если формат данных не поддерживается
        """
        content = self.prompt

        # Случай 1: Строка - это текстовый промпт
        if isinstance(content, str):
            self.type = "text"
            self.prompt_format = "string"
            return

        # Случай 2: Список - это чат-промпт
        if isinstance(content, list):
            self.type = "chat"
            # Пустой список - создаем пустой tuple_list
            if not content:
                self.prompt_format = "tuple_list"
                self.prompt = []
                return

            # Определяем формат по типу первого элемента и конвертируем в tuple_list
            first_item = content[0]
            if isinstance(first_item, dict):
                # Список словарей: конвертируем в список кортежей
                self.prompt = [(msg.get("role", "user"), msg.get("content", "")) for msg in content]
                self.prompt_format = "tuple_list"
            elif isinstance(first_item, list | tuple):
                # Уже список кортежей - оставляем как есть
                self.prompt_format = "tuple_list"
            else:
                # Неподдерживаемый формат элементов
                raise ValueError(f"Неподдерживаемый формат элементов списка промпта: {type(first_item)}")
            return

        # Неподдерживаемый тип данных
        raise ValueError(f"Неподдерживаемый тип данных промпта: {type(content)}")

    def format(self, target_format):
        """
        Преобразует промпт в указанный формат.

        Поддерживаемые форматы:
        - "string": строка (только для text или одиночного сообщения от user)
        - "dict_list": список словарей [{"role": "user", "content": "..."}]
        - "tuple_list": список кортежей [("user", "..."), ("assistant", "...")]

        Args:
            target_format: Целевой формат ("string", "dict_list" или "tuple_list")

        Returns:
            Промпт в указанном формате

        Raises:
            ValueError: Если формат не определен или преобразование невозможно
        """
        if self.prompt_format is None:
            raise ValueError("Формат промпта не определен")

        # Если формат уже совпадает - возвращаем копию без преобразования
        if self.prompt_format == target_format:
            if isinstance(self.prompt, str):
                return self.prompt
            if isinstance(self.prompt, list):
                return self.prompt.copy()
            return self.prompt

        # Для text промпта
        if self.type == "text":
            if target_format == "string":
                return self.prompt
            if target_format == "dict_list":
                return [{"role": "user", "content": self.prompt}]
            if target_format == "tuple_list":
                return [("user", self.prompt)]
            raise ValueError(f"Неизвестный целевой формат для text: {target_format}")

        # Для chat промпта (всегда хранится как tuple_list)
        if self.type == "chat":
            if target_format == "tuple_list":
                return self.prompt.copy()  # type: ignore
            if target_format == "dict_list":
                # Конвертируем из tuple_list в dict_list
                return [{"role": role, "content": content} for role, content in self.prompt]  # type: ignore
            if target_format == "langchain":
                messages = []
                for role, content in self.format("tuple_list"):
                    if role == "system":
                        msg_cls = SystemMessage
                    elif role == "user":
                        msg_cls = HumanMessage
                    elif role in ("assistant", "ai"):
                        msg_cls = AIMessage
                    else:
                        msg_cls = HumanMessage
                    # AICODE-NOTE: Return literal LangChain messages to avoid a second
                    # Jinja parse pass over already-compiled content.
                    messages.append(msg_cls(content=str(content)))
                return messages
            if target_format == "string":
                # Преобразуем в строку только если одно сообщение от user
                if len(self.prompt) == 1 and self.prompt[0][0] == "user":  # type: ignore
                    return str(self.prompt[0][1])  # type: ignore
                raise ValueError("Невозможно безопасно преобразовать историю чата в одну строку")
            raise ValueError(f"Неизвестный целевой формат для chat: {target_format}")

        raise ValueError(f"Неизвестный тип промпта: {self.type}")

    def __call__(self, target_format=None):
        """
        Возвращает промпт в указанном формате.

        Позволяет вызывать объект Prompt как функцию для получения промпта
        в нужном формате. Если формат не указан, возвращает в текущем формате.

        Args:
            target_format: Целевой формат ("string", "dict_list" или "tuple_list").
                          Если None, возвращает в текущем формате.

        Returns:
            Промпт в указанном или текущем формате
        """
        if target_format is None:
            # Возвращаем в текущем формате (копию для безопасности)
            if isinstance(self.prompt, str):
                return self.prompt
            if isinstance(self.prompt, list):
                return self.prompt.copy()
            return self.prompt
        return self.format(target_format)

    def compile(self, **variables):
        """
        Компилирует промпт, подставляя значения переменных.

        Заменяет переменные в формате {{variable_name}} на значения из variables.
        Работает как для текстовых, так и для чат-промптов.
        """

        def _substitute(text):
            """
            Заменяет переменные в тексте на их значения.
            """

            def repl(match):
                key = match.group(1)
                if key not in variables:
                    raise KeyError(f"Отсутствует значение для переменной '{key}'")
                return str(variables[key])

            return _VAR_PATTERN.sub(repl, text)

        if self.type == "text":
            text_prompt = self.format("string")
            if not isinstance(text_prompt, str):
                raise TypeError("Ошибка внутренней логики: для типа text формат должен быть string")
            self.prompt = _substitute(text_prompt)
            self.type = "text"
            self.prompt_format = "string"
            return self

        if self.type == "chat":
            if not isinstance(self.prompt, list):
                raise TypeError("Ошибка внутренней логики: для типа chat prompt должен быть списком")
            compiled_messages = []
            for role, content in self.prompt:  # type: ignore
                compiled_messages.append((role, _substitute(str(content))))
            self.prompt = compiled_messages
            self.type = "chat"
            self.prompt_format = "tuple_list"
            return self

        raise ValueError(f"Неизвестный тип prompt: {self.type}")

    def add_message(self, role, content):
        """
        Добавляет сообщение с указанной ролью в промпт.

        Если промпт типа text, конвертирует его в chat формат:
        - существующий текст становится system сообщением
        - добавляется новое сообщение

        Если промпт типа chat, добавляет сообщение в список.

        Args:
            role: Роль сообщения (например, "user", "assistant", "system")
            content: Содержимое сообщения
        """
        if self.type == "text":
            # Конвертируем text в chat: существующий текст становится system сообщением
            self.type = "chat"
            self.prompt = [("system", self.prompt)]
            self.prompt_format = "tuple_list"

        # Добавляем новое сообщение
        if not isinstance(self.prompt, list):
            raise ValueError(f"Для chat промпта prompt должен быть списком, получен {type(self.prompt)}")
        self.prompt.append((role, content))

        return self  # на случай перезаписи

    def add_user_message(self, content):
        """
        Добавляет user сообщение в промпт.

        Удобный метод для добавления сообщений от пользователя.

        Args:
            content: Содержимое сообщения
        """
        self.add_message("user", content)
        return self

    def add_assistant_message(self, content):
        """
        Добавляет assistant сообщение в промпт.

        Удобный метод для добавления сообщений от ассистента.

        Args:
            content: Содержимое сообщения
        """
        self.add_message("assistant", content)
        return self

    def add_messages(self, messages):
        """
        Добавляет несколько сообщений в промпт.

        Принимает список сообщений в том же формате, что и промпт:
        - для chat: список кортежей [("role", "content"), ...]

        Если промпт типа text, конвертирует его в chat формат:
        - существующий текст становится system сообщением
        - добавляются новые сообщения

        Args:
            messages: Список сообщений в формате [(role, content), ...]

        Raises:
            ValueError: Если формат сообщений некорректен
        """
        if not isinstance(messages, list):
            raise ValueError(f"Ожидается список сообщений, получен {type(messages)}")

        if self.type == "text":
            # Конвертируем text в chat: существующий текст становится system сообщением
            self.type = "chat"
            self.prompt = [("system", self.prompt)]
            self.prompt_format = "tuple_list"

        # Проверяем формат сообщений и добавляем их
        for msg in messages:
            if not isinstance(msg, list | tuple) or len(msg) != 2:
                raise ValueError(f"Каждое сообщение должно быть кортежем (role, content), получено: {msg}")
            role, content = msg
            self.prompt.append((role, content))
        return self

    def to_dict(self):
        """
        Преобразует объект Prompt в словарь.

        Returns:
            Словарь со всеми полями промпта
        """
        return {
            "name": self.name,
            "project": self.project,
            "type": self.type,
            "prompt": self.prompt,
            "prompt_format": self.prompt_format,
            "config": self.config,
            "version": self.version,
            "labels": self.labels,
            "tags": self.tags,
        }

    def to_file(self, filepath=None, overwrite=False):
        """
        Сохраняет промпт в YAML файл.

        Args:
            filepath: Путь к файлу. Если None, использует PROMPTS_DIR / f"{name}.yaml"
            overwrite: Разрешить перезапись существующего файла (по умолчанию False)

        Raises:
            ValueError: Если не указано имя промпта и путь к файлу
            FileExistsError: Если файл существует и overwrite=False
        """

        if filepath is None:
            if not self.name:
                raise ValueError("Не указано имя промпта и путь к файлу")
            filepath = PROMPTS_DIR / f"{self.name}.yaml"

        filepath = Path(filepath)

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"Файл уже существует: {filepath}. Используйте overwrite=True для перезаписи")

        data = {
            "name": self.name,
            "project": self.project or PROJECT_NAME,
        }

        if self.type == "text":
            data["prompt"] = self.format("string")
        elif self.type == "chat":
            data["prompt"] = self.format("dict_list")

        if self.config:
            data["config"] = self.config
        if self.labels:
            data["labels"] = self.labels
        if self.tags:
            data["tags"] = self.tags
        if self.version:
            data["version"] = self.version

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            yaml.dump(
                data,
                file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

        return filepath

    @classmethod
    def from_dict(cls, data):
        """
        Создает объект Prompt из словаря.

        Args:
            data: Словарь с данными промпта

        Returns:
            Объект Prompt
        """
        labels = data.get("labels")
        tags = data.get("tags")
        prompt = data.get("prompt", "")
        prompt_format = data.get("prompt_format")

        # Если это chat и пришел dict_list, конвертируем в tuple_list
        prompt_type = data.get("type")
        if prompt_type == "chat" and isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Конвертируем dict_list в tuple_list
            prompt = [(msg.get("role", "user"), msg.get("content", "")) for msg in prompt]
            prompt_format = "tuple_list"

        return cls(
            name=data["name"],
            project=data.get("project"),
            type=prompt_type,
            prompt=prompt,
            prompt_format=prompt_format,
            config=data.get("config"),
            version=data.get("version"),
            labels=list(labels) if labels else ["production"],
            tags=list(tags) if tags else [],
        )

    def send_to_langfuse(self):
        """
        Отправляет промпт в Langfuse.

        Создает или обновляет промпт в Langfuse с указанными параметрами.

        Returns:
            Результат создания промпта в Langfuse
        """
        prompt_name = qualify_prompt_name(self.name, self.project)
        result = langfuse.create_prompt(
            name=prompt_name,
            type=self.type,
            prompt=self.format("dict_list"),
            labels=self.labels,
            config={"project": self.project},
            tags=self.tags,
        )
        return result


# TODO: переработать и дополнить
def _validate_prompt_data(data, filename=None):
    """
    Валидирует данные промпта.
    """
    context = f" в файле '{filename}'" if filename else ""

    if "name" not in data:
        raise ValueError(f"Отсутствует поле 'name'{context}")

    if "/" in data["name"]:
        raise ValueError(f"Поле 'name' не должно содержать '/'{context}: {data['name']}")

    if "prompt" not in data and "messages" not in data:
        raise ValueError(f"Отсутствует поле 'prompt' или 'messages'{context}")

    content = data.get("prompt") or data.get("messages")
    if not isinstance(content, str | list):
        raise ValueError(f"Поле 'prompt' должно быть строкой или списком{context}, получен: {type(content)}")

    if isinstance(content, list):
        for elem in content:
            if isinstance(elem, dict):
                if "role" not in elem or "content" not in elem:
                    raise ValueError(f"Элемент списка должен содержать поля 'role' и 'content'{context}, получен: {elem}")
            elif isinstance(elem, list | tuple) and len(elem) == 2:
                if elem[0] not in ["user", "assistant", "system"]:
                    raise ValueError(
                        f"Роль в элементе списка должна быть 'user', 'assistant' или 'system'{context}, получен: {elem[0]}"
                    )
            else:
                raise ValueError(f"Элемент списка должен быть словарем или кортежем длиной 2{context}, получен: {type(elem)}")


def get_prompt_from_file(prompt_name=None, prompt_path=None):  # TODO: исправить логику передачи параметров
    """
    Загружает промпт из YAML файла.

    Ищет файл с именем {prompt_name}.yaml в директории промптов.
    Поддерживает альтернативное поле "messages" вместо "prompt".

    Args:
        prompt_name: Имя промпта (без расширения .yaml)

    Returns:
        Объект Prompt

    Raises:
        ValueError: Если файл пустой или не найден
        FileNotFoundError: Если файл не существует
    """
    assert prompt_name or prompt_path, "Необходимо указать prompt_name или prompt_path"
    prompt_path = prompt_path or PROMPTS_DIR / f"{prompt_name}.yaml"
    with open(prompt_path, encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Файл промпта пустой: {prompt_path}")

    _validate_prompt_data(data, filename=prompt_path.name)

    # Если имя не указано в файле, используем имя файла
    if not data.get("name"):
        logger.warning(
            "[PromptStore] Warning: в файле '%s' не указан 'name', использую '%s' как name",
            prompt_path,
            prompt_name,
        )
        data["name"] = prompt_name

    # Если проект не указан, используем проект по умолчанию
    if not PROJECT_NAME or PROJECT_NAME == "default":
        data["project"] = data.get("project", "default")

    if "prompt" not in data and "messages" in data:
        data["prompt"] = data["messages"]

    return Prompt.from_dict(data)


def _are_prompts_different(local_prompt: Prompt, remote_prompt: Prompt) -> bool:
    """
    Проверяет, отличаются ли локальный и удаленный промпты.
    Сравнивает тип, содержимое, конфиг, метки и теги.
    """
    if local_prompt.type != remote_prompt.type:
        return True

    if local_prompt.type == "text":
        if local_prompt.format("string") != remote_prompt.format("string"):
            return True
    elif local_prompt.type == "chat" and local_prompt.format("dict_list") != remote_prompt.format("dict_list"):
        return True

    if (local_prompt.config or {}) != (remote_prompt.config or {}):
        return True

    if set(local_prompt.labels) != set(remote_prompt.labels):
        return True
    return set(local_prompt.tags) != set(remote_prompt.tags)


# TODO: добавить update в файл
def get_prompt_from_langfuse(prompt_name, update_file=True):
    """
    Загружает промпт из Langfuse.

    Получает промпт по имени из Langfuse и преобразует в объект Prompt.
    Если проект не указан в данных, извлекает его из полного имени.

    Args:
        prompt_name: Имя промпта (может быть с проектом или без)
        update_file: Если True, обновляет (или создает) локальный файл промпта при отличии

    Returns:
        Объект Prompt

    Raises:
        AttributeError: Если объект от Langfuse не имеет нужных атрибутов
        KeyError: Если в данных отсутствуют обязательные поля
        ValueError: Если формат данных некорректен
    """
    lf_name = qualify_prompt_name(prompt_name)
    raw_prompt = langfuse.get_prompt(lf_name)
    prompt_data = raw_prompt.__dict__.copy()
    if "project" not in prompt_data or not prompt_data["project"]:
        if "/" not in lf_name:
            raise ValueError(f"Не удалось извлечь проект из имени промпта: {lf_name}")
        project, short_name = lf_name.split("/", 1)
        prompt_data["name"] = short_name
        prompt_data["project"] = project
    prompt = Prompt.from_dict(prompt_data)
    prompt.lf_raw = raw_prompt

    if update_file:
        try:
            prompt_name = prompt.name.split("/")[-1]
            filepath = PROMPTS_DIR / f"{prompt_name}.yaml"
            # Пытаемся загрузить локальный файл для сравнения
            try:
                local_prompt = get_prompt_from_file(prompt_name)

                if _are_prompts_different(local_prompt, prompt):
                    # Если отличаются, обновляем файл
                    prompt.to_file(filepath=filepath, overwrite=True)
                    logger.info("[PromptStore] Промпт '%s' обновлен в файле.", prompt_name)
            except (FileNotFoundError, ValueError):
                # Если файла нет или он пустой, создаем его
                prompt.to_file(filepath=filepath, overwrite=True)
                logger.info("[PromptStore] Файл для промпта '%s' создан.", prompt_name)

        except Exception as e:
            logger.exception("[PromptStore] Ошибка при обновлении файла промпта: %s", e)

    return prompt


def get_prompt(prompt_name, update_file=True):
    """
    Загружает промпт из Langfuse или из файла (fallback).

    Если LANGFUSE_ENABLED=False, загружает из локального файла напрямую.
    Иначе сначала пытается загрузить промпт из Langfuse.
    Если это не удается, загружает из локального YAML файла.

    Args:
        prompt_name: Имя промпта
        update_file: Если True, обновляет локальный файл при успешной загрузке из Langfuse

    Returns:
        Объект Prompt
    """
    if not LANGFUSE_ENABLED:
        return get_prompt_from_file(prompt_name)

    try:
        # Пытаемся загрузить из Langfuse
        return get_prompt_from_langfuse(prompt_name, update_file=update_file)
    except (AttributeError, KeyError, ValueError) as e:
        # Если не удалось, используем файл как резервный вариант
        logger.warning(
            "[PromptStore] Возникла ошибка при получении %s из Langfuse: %s, использую файл",
            prompt_name,
            e,
        )
        return get_prompt_from_file(prompt_name)


def upload_prompts_to_langfuse(prompts_dir):
    prompts_path = Path(prompts_dir)
    yaml_files = list(prompts_path.glob("*.yaml")) + list(prompts_path.glob("*.yml"))

    for yaml_file in yaml_files:
        try:
            prompt = get_prompt_from_file(prompt_path=yaml_file)
            prompt.send_to_langfuse()
            logger.info("%s/%s - отправлен в Langfuse", prompt.project, prompt.name)
        except Exception as e:
            logger.exception("[ОШИБКА] %s: %s", yaml_file.name, e)


class PromptStore(dict):
    def __getitem__(self, name):
        if name not in self:
            self[name] = get_prompt(name)
        return deepcopy(dict.__getitem__(self, name))


promptstore = PromptStore()

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from functools import wraps

from langfuse import observe, propagate_attributes

from src.common.config.config import settings

from .client import langfuse
from .prompt_kit import promptstore

PROJECT_NAME = settings.get("PROJECT_NAME", "default")


def _pick_mapping(args, kwargs):
    for value in args:
        if isinstance(value, Mapping):
            return value
    for value in kwargs.values():
        if isinstance(value, Mapping):
            return value
    return None


def _pick_sequence_of_mappings(args, kwargs):
    values = [*args, *kwargs.values()]
    for value in values:
        if not isinstance(value, Sequence):
            continue
        if isinstance(value, str | bytes | bytearray):
            continue

        items = list(value)
        if items and all(isinstance(item, Mapping) for item in items):
            return items
    return None


def _build_observation_input(fn, args, kwargs, in_keys):
    # Универсальная нормализация входа для Langfuse.
    # Почему так: разные функции в кодовой базе принимают payload по-разному (dict-state, обычные параметры,
    # или list[dict] как в kb_search). Хотим одинаковый UX в Langfuse без переписывания сигнатур повсюду.
    if not in_keys:
        return None

    state = _pick_mapping(args, kwargs)
    if state is not None:
        # 1) "Классический" случай: вход уже в Mapping (consultant_*), достаем указанные ключи.
        if len(in_keys) == 1:
            return state.get(in_keys[0])
        return {key: state.get(key) for key in in_keys}

    try:
        # 2) Обычные параметры: привязываем args/kwargs к именам параметров функции.
        # Это покрывает случаи вроде web_search_async(query=..., top_k=..., domains=...).
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except TypeError:
        bound = None

    if bound is not None:
        bound.apply_defaults()
        if any(key in bound.arguments for key in in_keys):
            if len(in_keys) == 1:
                return bound.arguments.get(in_keys[0])
            return {key: bound.arguments.get(key) for key in in_keys}

    # 3) kb_search: payload прокидывается как list[dict], а не как именованные параметры.
    # AICODE-NOTE: Это эвристика. Если у функции несколько аргументов вида list[dict], выбор может быть не тем.
    sequence_state = _pick_sequence_of_mappings(args, kwargs)
    if sequence_state is not None:
        if len(in_keys) == 1:
            return [item.get(in_keys[0]) for item in sequence_state]
        return [{key: item.get(key) for key in in_keys} for item in sequence_state]
    return None


def span(as_type="span", name=None, in_keys=(), out_keys=(), prompt_name=None):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            prompt_obj = None
            if as_type == "generation" and prompt_name is not None:
                try:
                    p = promptstore[prompt_name]
                    if p.lf_raw is not None:
                        prompt_obj = p.lf_raw
                except Exception:
                    prompt_obj = None

            with langfuse.start_as_current_observation(
                as_type=as_type,
                name=name or fn.__name__,
                prompt=prompt_obj if prompt_obj is not None else None,
            ) as obs:
                if in_keys:
                    payload = _build_observation_input(fn, args, kwargs, in_keys)
                    if payload is not None:
                        obs.update(input=payload)
                # Если не указаны in_keys, логируем первый аргумент как есть
                elif args:
                    obs.update(input=args[0] if len(args) == 1 else args)

                result = fn(*args, **kwargs)

                # Обработка output
                if out_keys:
                    if isinstance(result, dict):
                        if len(out_keys) == 1:
                            obs.update(output=result.get(out_keys[0]))
                        else:
                            obs.update(output={key: result.get(key) for key in out_keys})
                # Если не указаны out_keys, логируем результат как есть
                elif result is not None:
                    obs.update(output=result)

                return result

        return wrapped

    return deco


def aspan(as_type="span", name=None, in_keys=(), out_keys=(), prompt_name=None):
    def deco(fn):
        @wraps(fn)
        async def wrapped(*args, **kwargs):
            prompt_obj = None
            if as_type == "generation" and prompt_name is not None:
                try:
                    p = promptstore[prompt_name]
                    if p.lf_raw is not None:
                        prompt_obj = p.lf_raw
                except Exception:
                    prompt_obj = None

            with langfuse.start_as_current_observation(
                as_type=as_type,
                name=name or fn.__name__,
                prompt=prompt_obj if prompt_obj is not None else None,
            ) as obs:
                if in_keys:
                    payload = _build_observation_input(fn, args, kwargs, in_keys)
                    if payload is not None:
                        obs.update(input=payload)
                elif args:
                    obs.update(input=args[0] if len(args) == 1 else args)

                result = await fn(*args, **kwargs)

                if out_keys:
                    if isinstance(result, dict):
                        if len(out_keys) == 1:
                            obs.update(output=result.get(out_keys[0]))
                        else:
                            obs.update(output={key: result.get(key) for key in out_keys})
                elif result is not None:
                    obs.update(output=result)

                return result

        return wrapped

    return deco


def root(
    name: str | None = PROJECT_NAME,
    get_user_id=lambda *a, **k: k.get("user_id"),
    get_session_id=lambda *a, **k: k.get("session_id"),
    static_tags: list[str] | None = None,
    static_metadata: dict | None = None,
    version: str | None = None,
):
    """
    Декоратор для корневой функции пайплайна.

    Ожидаем, что в аргументах функции будут user_id и session_id
    (или свои геттеры через get_user_id/get_session_id).
    Внутри:
      - создаёт trace (@observe)
      - вешает propagate_attributes(...) на весь пайплайн
    """
    static_tags = static_tags or []
    static_metadata = static_metadata or {}

    def deco(fn):
        # через observe создаём trace + root-observation
        @observe(name=name or fn.__name__)
        @wraps(fn)
        def wrapped(*args, **kwargs):
            user_id = get_user_id(*args, **kwargs)
            session_id = get_session_id(*args, **kwargs)

            # собираем всё, что хотим прокинуть
            propagate_kwargs = {
                "metadata": static_metadata,
                "tags": static_tags,
            }
            if user_id is not None:
                propagate_kwargs["user_id"] = user_id
            if session_id is not None:
                propagate_kwargs["session_id"] = session_id
            if version is not None:
                propagate_kwargs["version"] = version

            # если нечего прокидывать — просто пустой контекст
            ctx = propagate_attributes(**propagate_kwargs) if propagate_kwargs else nullcontext()

            with ctx:
                result = fn(*args, **kwargs)
                return result

        return wrapped

    return deco

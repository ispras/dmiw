import argparse
import json
from pydantic import ValidationError
from .settings import WatermarkSettings
from .tasks import embed_task, extract_task
from .algorithms import ALGORITHM_SETTINGS_REGISTRY
from typing import get_args, get_origin, Literal, Union
from pathlib import Path
from pydantic_core import PydanticUndefined
import math
from numbers import Number


def round_to_n_significant(x: float, n: int = 3) -> float:
    """Округляет число x до n значащих знаков."""
    if x == 0:
        return 0.0
    # Для отрицательных чисел работаем с модулем
    sign = 1 if x > 0 else -1
    abs_x = abs(x)
    # Количество знаков до запятой (в степени 10)
    order = math.floor(math.log10(abs_x))
    # Округляем до нужного десятичного разряда
    precision = n - 1 - order
    rounded = round(abs_x, precision)
    return sign * rounded


def round_metrics(data, n: int = 3):
    """Рекурсивно округляет все числа в структуре data до n значащих знаков."""
    if isinstance(data, dict):
        return {k: round_metrics(v, n) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_metrics(item, n) for item in data]
    elif isinstance(data, tuple):
        return tuple(round_metrics(item, n) for item in data)
    elif isinstance(data, Number):
        # Применяем округление только к float (int оставляем как есть, если нужно)
        if isinstance(data, float):
            return round_to_n_significant(data, n)
        else:
            return data
    else:
        return data


def add_pydantic_model_to_parser(parser, model_class):
    """Автоматически добавляет аргументы в argparse из полей Pydantic модели."""
    for field_name, field_info in model_class.model_fields.items():
        if field_name == "algorithm_name":
            continue  # Это поле определяется выбором подпарсера

        arg_name = f"--{field_name}"
        kwargs = {
            "help": field_info.description or "",
        }

        # Определяем тип аргумента для argparse
        field_type = field_info.annotation

        # Обработка Optional (если есть)
        if get_origin(field_type) is Union:
            # Берем первый тип, отбрасывая NoneType
            field_type = [
                t for t in get_args(field_type) if t is not type(None)
            ][0]

        # Обработка Literal (для choices в argparse)
        if get_origin(field_type) is Literal:
            kwargs["choices"] = get_args(field_type)
            kwargs["type"] = type(
                get_args(field_type)[0]
            )  # тип первого элемента
        elif field_type is float:
            kwargs["type"] = float
        elif field_type is int:
            kwargs["type"] = int
        elif field_type is str or field_type is Path:
            kwargs["type"] = str
        elif field_type is bool:
            kwargs["action"] = "store_true"

        # Обработка значений по умолчанию
        if not field_info.is_required():
            # Если поле не обязательное, берем дефолт
            if field_info.default is not PydanticUndefined:
                kwargs["default"] = field_info.default
        else:
            kwargs["required"] = True

        parser.add_argument(arg_name, **kwargs)


def extract_algo_settings_from_args(args):
    """Автоматически собирает словарь настроек алгоритма из args."""
    algo_name = args.algorithm
    if not algo_name:
        return {}

    settings_class = ALGORITHM_SETTINGS_REGISTRY[algo_name]
    algo_dict = {"algorithm_name": algo_name}

    # Собираем только те поля, которые есть в модели
    for field_name in settings_class.model_fields.keys():
        if field_name == "algorithm_name":
            continue
        if hasattr(args, field_name):
            algo_dict[field_name] = getattr(args, field_name)

    return algo_dict


def create_algorithm_parser(subparsers, algorithm_name: str, help_text: str):
    """Создать парсер для конкретного алгоритма."""
    algo_parser = subparsers.add_parser(
        algorithm_name,
        help=help_text,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    settings_class = ALGORITHM_SETTINGS_REGISTRY[algorithm_name]
    add_pydantic_model_to_parser(algo_parser, settings_class)
    return algo_parser


def main():
    parser = argparse.ArgumentParser(
        description="Утилита для внедрения и извлечения ЦВЗ"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Доступные команды"
    )

    # === Embed парсер ===
    embed_parser = subparsers.add_parser(
        "embed", help="Внедрить ЦВЗ в изображение"
    )
    embed_parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Путь к исходному изображению",
    )
    embed_parser.add_argument(
        "-k",
        "--key",
        default="deadbeaf",
        type=str,
        help="Ключ ЦВЗ в формате hex строки",
    )

    embed_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Путь для сохранения изображения с ЦВЗ",
    )

    # Подпарсеры для алгоритмов внутри embed
    algo_subparsers_embed = embed_parser.add_subparsers(
        dest="algorithm", help="Алгоритм внедрения"
    )

    extract_parser = subparsers.add_parser(
        "extract", help="Извлечь ЦВЗ из изображения"
    )
    extract_parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Путь к изображению с ЦВЗ",
    )
    extract_parser.add_argument(
        "-k",
        "--key",
        default="deadbeaf",
        type=str,
        help="Оригинальный ключ ЦВЗ в формате hex строки для подсчета метрик",
    )

    # Подпарсеры для алгоритмов внутри extract
    algo_subparsers_extract = extract_parser.add_subparsers(
        dest="algorithm", help="Алгоритм извлечения"
    )

    # Создаём парсеры для каждого алгоритма
    for algo_name, algo_settings_class in ALGORITHM_SETTINGS_REGISTRY.items():
        create_algorithm_parser(
            algo_subparsers_embed,
            algo_name,
            f"Использовать алгоритм {algo_name}",
        )
        create_algorithm_parser(
            algo_subparsers_extract,
            algo_name,
            f"Использовать алгоритм {algo_name}",
        )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Создаём словарь настроек алгоритма
    algorithm_dict = extract_algo_settings_from_args(args)

    try:
        settings = WatermarkSettings(algorithm=algorithm_dict)
    except ValidationError as e:
        print("Ошибка валидации настроек:")
        print(e)
        return

    if args.command == "embed":
        metrics = embed_task(
            image_path=args.image,
            key_hex=args.key,
            output_path=args.output,
            settings=settings,
        )
    elif args.command == "extract":
        metrics = extract_task(
            image_path=args.image, key_hex=args.key, settings=settings
        )
        print("Результаты извлечения:")
    metrics = round_metrics(metrics, n=3)
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()

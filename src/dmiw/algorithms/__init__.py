from typing import Dict, Type, Union
from .base import BaseAlgorithmSettings
from .trustmark import TrustMark, TrustMarkSettings
from .videoseal import VideoSeal, VideoSealSettings
from .invismark import InvisMark, InvisMarkSettings

# Реестр классов настроек
ALGORITHM_SETTINGS_REGISTRY: Dict[str, Type[BaseAlgorithmSettings]] = {
    "trustmark": TrustMarkSettings,
    "videoseal": VideoSealSettings,
    "invismark": InvisMarkSettings,
}

# Реестр классов алгоритмов
ALGORITHM_REGISTRY: Dict[str, Type] = {
    "trustmark": TrustMark,
    "videoseal": VideoSeal,
    "invismark": InvisMark,
}

# Тип для дискриминируемого объединения
AlgorithmSettings = Union[TrustMarkSettings, VideoSealSettings]


def get_algorithm(settings: BaseAlgorithmSettings):
    """Получить экземпляр алгоритма по настройкам."""
    algorithm_name = settings.algorithm_name
    if algorithm_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Алгоритм {algorithm_name} не найден")

    # Создаём экземпляр алгоритма, передавая настройки
    algorithm_class = ALGORITHM_REGISTRY[algorithm_name]
    return algorithm_class(
        **settings.dict(exclude={"algorithm_name", "key_length"})
    )


def settings2capacity(settings: BaseAlgorithmSettings):
    if settings.algorithm_name == "trustmark":
        return 100
    elif settings.algorithm_name == "videoseal":
        settings: VideoSealSettings
        if settings.model == "pixelseal":
            return 256
        elif settings.model == "isp_128":
            return 128
    elif settings.algorithm_name == "invismark":
        return 100

    raise ValueError(f"Неизвестный алгортим: {settings.algorithm_name}")

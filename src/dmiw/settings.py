from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Union
from .algorithms.trustmark import TrustMarkSettings
from .algorithms.videoseal import VideoSealSettings
from .algorithms.invismark import InvisMarkSettings

# Дискриминируемое объединение по полю algorithm_name
AlgorithmSettings = Union[
    TrustMarkSettings, VideoSealSettings, InvisMarkSettings
]


class WatermarkSettings(BaseSettings):
    """Настройки пакета с поддержкой разных алгоритмов."""

    model_config = SettingsConfigDict(
        env_prefix="DMIW_", env_file=".env", extra="ignore"
    )

    # Используем дискриминируемое объединение
    algorithm: AlgorithmSettings = Field(
        ...,
        discriminator="algorithm_name",
        description="Настройки алгоритма внедрения ЦВЗ",
    )

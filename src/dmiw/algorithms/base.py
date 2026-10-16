from pydantic import BaseModel, Field
from typing import Literal
from abc import ABC, abstractmethod
import torch


class BaseAlgorithmSettings(BaseModel):
    """Базовый класс для настроек алгоритмов."""

    algorithm_name: str
    wm_strength: float = Field(1, description="Коэффициент силы внедрения")
    device: str = Field(
        "cuda:0" if torch.cuda.is_available() else "cpu",
        description="Устройство для запуска алгоритма, поддерживается формат библиотеки torch",
    )
    interpolation_before: Literal[
        "nearest", "bilinear", "bicubic", "lanczos", "area", "nearest-exact"
    ] = Field(
        "bilinear",
        description="Метод интерполяции при приведении входного изображения в рабочее разрешение метода",
    )
    interpolation_after: Literal[
        "nearest", "bilinear", "bicubic", "lanczos", "area", "nearest-exact"
    ] = Field(
        "bilinear",
        description="Метод интерполяции при приведении накладываемого изображения ЦВЗ в исходное разрешения входного изображения",
    )


class BaseAlgorithm(ABC):
    """Базовый класс для алгоритмов ЦВЗ."""

    @abstractmethod
    def embed(self, image: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Внедрение ЦВЗ в изображение."""
        pass

    @abstractmethod
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """Извлечение ЦВЗ из изображения."""
        pass

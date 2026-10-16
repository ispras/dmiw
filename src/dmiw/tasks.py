import torch
import torchvision
from time import perf_counter
from scipy.stats import binomtest
from .algorithms import get_algorithm, settings2capacity
from .settings import WatermarkSettings
from .metrics import lpips, ssim, psnr


def hex_str_to_tensor(
    hex_str: str, max_key_length_bits: int = None
) -> torch.Tensor:
    """Конвертация hex строки в тензор бит (0.0 и 1.0) с проверкой длины."""
    # Убираем возможный префикс '0x' и переводим в нижний регистр
    hex_str_clean = hex_str.lower().replace("0x", "")
    if not hex_str_clean:
        raise ValueError("Ключ не может быть пустым")

    try:
        # Конвертируем hex в бинарную строку.
        # zfill добивает нулями слева: 1 hex символ = ровно 4 бита
        bits_str = bin(int(hex_str_clean, 16))[2:].zfill(
            len(hex_str_clean) * 4
        )
    except ValueError:
        raise ValueError("Неверный формат hex строки для ключа")

    bits = [int(b) for b in bits_str]

    # Проверка на ограничение длины ключа
    if max_key_length_bits is not None and len(bits) > max_key_length_bits:
        max_hex_len = max_key_length_bits // 4
        print(
            f"ВНИМАНИЕ: Длина поданного ключа ({len(hex_str_clean)} 16-ричных символов) "
            f"превышает максимально допустимую для алгоритма ( {max_hex_len} 16-ричных символов). "
            f"Ключ будет обрезан до допустимого размера!"
        )
        bits = bits[:max_key_length_bits]
    elif max_key_length_bits is not None and len(bits) < max_key_length_bits:
        bits = (max_key_length_bits - len(bits)) * [0] + bits
    return torch.tensor(bits, dtype=torch.float32)


def tensor_to_hex_str(tensor: torch.Tensor) -> str:
    """Конвертация тензора бит (0.0 и 1.0) в hex строку."""
    # Конвертируем тензор в строку из '0' и '1'
    bits_str = "".join([str(int(b)) for b in tensor.flatten().tolist()])

    if not bits_str:
        return ""

    # Длина бинарной строки должна быть кратна 4 для корректного перевода в hex.
    # Добиваем нулями слева, если длина не кратна 4 (например, при обрезании ключа)
    padding_len = (4 - len(bits_str) % 4) % 4
    bits_str_padded = bits_str.zfill(len(bits_str) + padding_len)

    # Переводим бинарную строку в целое число, затем в hex
    hex_val = hex(int(bits_str_padded, 2))[2:]  # Отрезаем префикс '0x'

    # Добиваем hex строку нулями слева, чтобы она соответствовала количеству бит
    expected_hex_len = len(bits_str_padded) // 4
    hex_str = hex_val.zfill(expected_hex_len)

    return hex_str


def load_image(path: str) -> torch.Tensor:
    """Загрузка изображения через torchvision."""
    img = torchvision.io.read_image(
        path, mode=torchvision.io.ImageReadMode.RGB
    )
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    return img


def save_image(tensor: torch.Tensor, path: str):
    """Сохранение тензора в файл."""
    torchvision.utils.save_image(tensor, path)


def calculate_metrics(
    original_key: torch.Tensor, extracted_key: torch.Tensor
) -> dict:
    """Вычисление BER и p-value."""
    orig = original_key.int()
    extr = extracted_key.int()

    errors = (orig != extr).sum().item()
    ber = errors / len(orig)

    matches = len(orig) - errors
    try:
        p_val = binomtest(
            matches, n=len(orig), p=0.5, alternative="greater"
        ).pvalue
    except ImportError:
        from scipy.stats import binom_test

        p_val = binom_test(matches, n=len(orig), p=0.5)

    return {"BER": ber, "P-value": p_val}


def embed_task(
    image_path: str,
    key_hex: str,
    output_path: str,
    settings: WatermarkSettings,
) -> dict:
    """Таска внедрения ЦВЗ."""
    image = load_image(image_path)

    key_length = settings2capacity(settings.algorithm)
    # Передаем ограничение из настроек алгоритма
    key = hex_str_to_tensor(key_hex, max_key_length_bits=key_length)
    s_time = perf_counter()
    algo = get_algorithm(settings.algorithm)
    init_time = perf_counter() - s_time

    s_time = perf_counter()
    watermarked_image = algo.embed(image, key)
    embed_time = perf_counter() - s_time

    save_image(watermarked_image, output_path)
    print(f"ЦВЗ успешно внедрён. Изображение сохранено в {output_path}")
    return {
        "Внедренное сообщение": tensor_to_hex_str(key),
        "SSIM": ssim(image, watermarked_image),
        "PSNR": psnr(image, watermarked_image),
        "LPIPS": lpips(image, watermarked_image),
        "Время инициализации": init_time,
        "Время внедрения": embed_time,
    }


def extract_task(
    image_path: str, key_hex: str, settings: WatermarkSettings
) -> dict:
    """Таска извлечения ЦВЗ и подсчета метрик."""
    image = load_image(path=image_path)
    key_length = settings2capacity(settings.algorithm)
    # Передаем ограничение из настроек алгоритма
    original_key = hex_str_to_tensor(key_hex, max_key_length_bits=key_length)

    s_time = perf_counter()
    algo = get_algorithm(settings.algorithm)
    init_time = perf_counter() - s_time

    # Извлекаем ровно столько бит, сколько было внедрено
    s_time = perf_counter()
    extracted_key = algo.extract(image)
    extract_time = perf_counter() - s_time
    metrics = {
        "Сообщение для сравнения": tensor_to_hex_str(original_key),
        "Извлеченное сообщение  ": tensor_to_hex_str(extracted_key),
    }
    metrics.update(calculate_metrics(original_key, extracted_key))
    metrics.update(
        {
            "Время инициализации": init_time,
            "Время извлечения": extract_time,
        }
    )
    return metrics

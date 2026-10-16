# DMIW — Deep Methods for Image Watermarking

Реализация утилиты командной строки для применения нейросетевых методов маркирования изображений.

## Реализованные алгоритмы

<table class="comparison-table">
  <thead>
    <tr>
      <th>Алгоритм</th>
      <th>Модель</th>
      <th>Длина ключа (hex)</th>
      <th>Описание</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Trustmark</td>
      <td>Q</td>
      <td>25</td>
      <td>Основная модель от авторов. Баланс устойчивости к атакам и незаметности.</td>
    </tr>
    <tr>
      <td>P</td>
      <td>25</td>
      <td>Модель с повышенной незаметностью за счёт сниженной устойчивости (авторская).</td>
    </tr>
    <tr>
      <td rowspan="2">Invismark</td>
      <td>invismark</td>
      <td>25</td>
      <td>Исходная модель Invismark от авторов.</td>
    </tr>
    <tr>
      <td>isp</td>
      <td>25</td>
      <td>Модель, обученная в ИСП РАН на основе исходного кода Invismark.</td>
    </tr>
    <tr>
      <td rowspan="2">Videoseal</td>
      <td>pixelseal</td>
      <td>64</td>
      <td>Оригинальный Pixelseal. Высокая незаметность и устойчивость.</td>
    </tr>
    <tr>
      <td>isp_128</td>
      <td>32</td>
      <td>Адаптация ИСП РАН на базе кода Videoseal. Меньшая емкость, более быстрое внедрение.</td>
    </tr>
  </tbody>
</table>

## Установка

Рекомендуемое окружение:

* ОС Ubuntu 22.04 LTS
* Python 3.10

Установка системных библиотек:

```
# apt install python3-pip python3.10-venv
```

Активация окружения:

```
python3 -m venv venv
source venv/bin/activate
```

Установка проекта в окружение:

```
pip install -e .
```

Загрузка весов моделей:

```
python scripts/download_models.py --all
```

Более подробно загрузка моделей описана в `models/README.md`.

## Запуск

Утилита позволяет внедрять цифровые водяные знаки в изображения, а также производить процедуру извлечения.

### Внедрение ЦВЗ

Команда запуска процедуры внедрения:

```bash
dmiw embed -i <путь/к/входному/изображению> -o <путь/для/сохранения/результата> -k <сообщение-ключ для внедрения в hex формате> {алгоритм} [параметры алгоритма]
```

Пример:

```bash
dmiw embed -i Lenna.png -o result.png -k deadbeaf123 trustmark --wm_strength 1 --model P

    ЦВЗ успешно внедрён. Изображение сохранено в result.png
    {
        "Внедренное сообщение": "00000000000000deadbeaf123",
        "SSIM": 0.998,
        "PSNR": 48.0,
        "LPIPS": 0.000758,
        "Время инициализации": 0.332,
        "Время внедрения": 0.0618
    }
```

Справка по команде внедрения:

```bash
dmiw embed -h

usage: dmiw embed [-h] -i IMAGE [-k KEY] -o OUTPUT {trustmark,videoseal,invismark} ...

    positional arguments:
    {trustmark,videoseal,invismark}
                            Алгоритм внедрения
        trustmark           Использовать алгоритм trustmark
        videoseal           Использовать алгоритм videoseal
        invismark           Использовать алгоритм invismark

    options:
    -h, --help            show this help message and exit
    -i IMAGE, --image IMAGE
                            Путь к исходному изображению
    -k KEY, --key KEY     Ключ ЦВЗ в формате hex строки
    -o OUTPUT, --output OUTPUT
                            Путь для сохранения изображения с ЦВЗ
```

Справка по параметрам алгоритма:

```bash
dmiw embed trustmark -h
    usage: dmiw embed trustmark [-h] [--wm_strength WM_STRENGTH] [--device DEVICE] [--interpolation_before {nearest,bilinear,bicubic,lanczos,area,nearest-exact}]
                                [--interpolation_after {nearest,bilinear,bicubic,lanczos,area,nearest-exact}] [--model_folder MODEL_FOLDER] [--model {P,Q}]

    options:
    -h, --help            show this help message and exit
    --wm_strength WM_STRENGTH
                            Коэффициент силы внедрения (default: 1)
    --device DEVICE       Устройство для запуска алгоритма, поддерживается формат библиотеки torch (default: cpu)
    --interpolation_before {nearest,bilinear,bicubic,lanczos,area,nearest-exact}
                            Метод интерполяции при приведении входного изображения в рабочее разрешение метода (default: bilinear)
    --interpolation_after {nearest,bilinear,bicubic,lanczos,area,nearest-exact}
                            Метод интерполяции при приведении накладываемого изображения ЦВЗ в исходное разрешения входного изображения (default: bilinear)
    --model_folder MODEL_FOLDER
                            Путь к директории с весами моделей метода Trustmark (default: /home/aleksey/image_watermarking/dmiw/models/trustmark)
    --model {P,Q}         Тип модели алгоритма TrustMark (default: P)

```

### Извлечение ЦВЗ

Команда запуска процедуры извлечения:

```bash
dmiw extract -i <путь/к/входному/изображению> -k <сообщение-ключ для сравнения с извлеченным в hex формате> {алгоритм} [параметры алгоритма]
```

Пример:

```bash
dmiw extract -i result.png -k deadbeaf123 trustmark --wm_strength 1 --model P
    
    Результаты извлечения:
    {
        "Сообщение для сравнения": "00000000000000deadbeaf123",
        "Извлеченное сообщение  ": "00000000000000deadbeaf123",
        "BER": 0.0,
        "P-value": 1.5800000000000003e-30,
        "Время инициализации": 0.351,
        "Время извлечения": 0.0355
    }
```

Справка по команде извлечения:

```bash
dmiw extract -h

    usage: dmiw extract [-h] -i IMAGE [-k KEY] {trustmark,videoseal,invismark} ...

    positional arguments:
    {trustmark,videoseal,invismark}
                            Алгоритм извлечения
        trustmark           Использовать алгоритм trustmark
        videoseal           Использовать алгоритм videoseal
        invismark           Использовать алгоритм invismark

    options:
    -h, --help            show this help message and exit
    -i IMAGE, --image IMAGE
                            Путь к изображению с ЦВЗ
    -k KEY, --key KEY     Оригинальный ключ ЦВЗ в формате hex строки для подсчета метрик
```

Справка по параметрам алгоритма:

```bash
dmiw extract trustmark -h

    usage: dmiw extract trustmark [-h] [--wm_strength WM_STRENGTH] [--device DEVICE] [--interpolation_before {nearest,bilinear,bicubic,lanczos,area,nearest-exact}]
                                [--interpolation_after {nearest,bilinear,bicubic,lanczos,area,nearest-exact}] [--model_folder MODEL_FOLDER] [--model {P,Q}]

    options:
    -h, --help            show this help message and exit
    --wm_strength WM_STRENGTH
                            Коэффициент силы внедрения (default: 1)
    --device DEVICE       Устройство для запуска алгоритма, поддерживается формат библиотеки torch (default: cpu)
    --interpolation_before {nearest,bilinear,bicubic,lanczos,area,nearest-exact}
                            Метод интерполяции при приведении входного изображения в рабочее разрешение метода (default: bilinear)
    --interpolation_after {nearest,bilinear,bicubic,lanczos,area,nearest-exact}
                            Метод интерполяции при приведении накладываемого изображения ЦВЗ в исходное разрешения входного изображения (default: bilinear)
    --model_folder MODEL_FOLDER
                            Путь к директории с весами моделей метода Trustmark (default: /home/aleksey/image_watermarking/dmiw/models/trustmark)
    --model {P,Q}         Тип модели алгоритма TrustMark (default: P)
```

## Метрики

### Метрики внедрения

* **Внедренное сообщение** - ключ-сообщение в hex-формате (16-ричный, 4 бита на символ), дополняется или обрезается по емкости метода
* **SSIM** - значение метрики незаметности SSIM, может быть в диапозоне от -1 до 1, значения **больше** 0.99 говорят о высокой незаметности
* **PSNR**- значение метрики незаметности PSNR, значения **больше** 42 говорят о высокой незаметности
* **LPIPS** - значение метрики незаметности LPIPS, значения **ниже** 0.01 говорят о высокой незаметности
* **Время инициализации** - время, затраченное на инициализацию модели алгоритма внедрения (в секундах). В основном - загрузка весов модели
* **Время внедрения** - время обработки одного изображения (в секундах)

### Метрики извлечения

* **Сообщение для сравнения** - ключ-сообщение в hex-формате (16-ричный, 4 бита на символ), переданное пользователем, приведенное к размеру ключа алгоритма
* **Извлеченное сообщение** -  ключ-сообщение в hex-формате (16-ричный, 4 бита на символ), извлеченное из изображения
* **BER** (Bit Error Rate) - доля неверно извлеченных бит (сравнение внедренного и извлеченного сообщений, бинарный формат). Близкие к 0 значения говорят о высокой вероятности наличия ЦВЗ в изображении
* **P-value** - значение p-value. Показывает вероятность получить настолько же близкий извлеченный ключ-сообщение, если взять случайное значение. Значения ниже 10e-6 говорят о высокой уверенности, что в изображение был внедрен заданный ключ
* **Время инициализации** - время, затраченное на инициализацию модели алгоритма извлечения (в секундах). В основном - загрузка весов модели
* **Время извлечения** - время обработки одного изображения (в секундах)
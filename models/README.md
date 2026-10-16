# Веса моделей

Эта директория предназначена для весов и конфигураций моделей TrustMark, InvisMark
и VideoSeal, используемых в DMIW. Файлы доступны в
[публичной папке Nextcloud](https://nextcloud.ispras.ru/index.php/s/zM5xXfiz3GpSytF).

Для автоматической загрузки используется скрипт
[`scripts/download_models.py`](../scripts/download_models.py).
Требуется Python 3.10 или новее. Дополнительные библиотеки для скачивания
устанавливать не нужно.

Все команды ниже выполняются **из корня репозитория**.

## Загрузка

Скачать все модели и соответствующие YAML-конфигурации:

```bash
python scripts/download_models.py --all
```

Общий размер файлов составляет около **4,37 ГБ**. При необходимости можно скачать
только одну модель, например TrustMark P:

```bash
python scripts/download_models.py --model trustmark-p
```

Для загрузки нескольких моделей перечислите их имена через пробел:

```bash
python scripts/download_models.py --model trustmark-p trustmark-q pixelseal
```

Посмотреть список доступных моделей без скачивания:

```bash
python scripts/download_models.py --list
```

## Доступные модели

Имена в столбце «Модель» используются в аргументе `--model` скрипта загрузки.
Столбцы «Алгоритм DMIW» и «Вариант DMIW» соответствуют выбору алгоритма
и параметру `--model` при внедрении и извлечении водяного знака.

| Модель | Алгоритм DMIW | Вариант DMIW | Размер весов, МБ |
|---|---|---|---:|
| `trustmark-p` | `trustmark` | `P` | 64,95 |
| `trustmark-q` | `trustmark` | `Q` | 64,95 |
| `invismark` | `invismark` | `invismark` | 1233,91 |
| `invismark-isp` | `invismark` | `isp` | 1233,91 |
| `pixelseal` | `videoseal` | `pixelseal` | 1237,43 |
| `videoseal-isp-128` | `videoseal` | `isp_128` | 533,22 |

По умолчанию файлы сохраняются в поддиректории `models/trustmark/`,
`models/invismark/` и `models/videoseal/`. Вместе с весами скачиваются необходимые
YAML-конфигурации. Полный набор содержит восемь файлов весов и восемь конфигураций.

## Проверка файлов

При скачивании скрипт проверяет размер и контрольную сумму SHA-256 каждого файла.
Ожидаемые значения указаны в [`weights.json`](weights.json).

Уже скачанные файлы с корректными размером и контрольной суммой пропускаются.
Повторный запуск позволяет загрузить отсутствующие или заменить повреждённые
файлы. Новый файл заменяет существующий только после успешной проверки.

Проверить все локальные файлы без обращения к сети:

```bash
python scripts/download_models.py --verify
```

Проверить только выбранную модель:

```bash
python scripts/download_models.py --model trustmark-p --verify
```

Если проверяемые файлы отсутствуют или повреждены, скрипт завершится с кодом `1`.
В режиме `--verify` файлы не скачиваются и не изменяются.

## Другой каталог хранения

Чтобы сохранить модели вне репозитория, укажите каталог через `--dest`:

```bash
python scripts/download_models.py --model trustmark-p --dest "$HOME/.cache/dmiw"
```

В этом примере веса и конфигурация TrustMark P будут сохранены в
`~/.cache/dmiw/trustmark/`. При запуске DMIW передайте этот путь через
`--model_folder`:

```bash
dmiw embed -i input.png -o watermarked.png -k deadbeaf123 \
  trustmark --model P --model_folder "$HOME/.cache/dmiw/trustmark"
```

Для проверки файлов в другом каталоге используйте те же параметры:

```bash
python scripts/download_models.py --model trustmark-p \
  --dest "$HOME/.cache/dmiw" --verify
```

## Параметры загрузки

По умолчанию скрипт выполняет до трёх попыток загрузки каждого файла.
Тайм-аут сетевых операций составляет 60 секунд. Эти значения можно изменить:

```bash
python scripts/download_models.py --model trustmark-p --attempts 5 --timeout 120
```

После прерывания незавершённый файл скачивается заново. Файлы, загрузка и проверка
которых уже завершились, повторно не скачиваются.

Адрес Nextcloud уже указан в `weights.json`. Для использования другой публичной
папки с такой же структурой можно передать её адрес через `--base-url` или
обновить поле `base_url` в этом файле.

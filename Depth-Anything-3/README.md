# Quick Start

Заходим в папку da3_streaming:
```bash
cd da3_streaming
```

Загружаем нужные веса (на cds2 и aicenter веса уже скачаны, поэтому переходим к следующему шагу):
```bash
bash ./scripts/download_weights.sh
```

Собираем контейнер:
```bash
bash build.sh
```

По умолчанию параметры чанков подбираются автоматически по числу кадров:
```text
chunk_size = 350, если кадров больше 350, иначе number_of_frames - 1
overlap = chunk_size // 2
loop_chunk_size = chunk_size // 2
```

Если нужно использовать значения напрямую из `da3_streaming/configs/base_config.yaml`, запускаем скрипт с флагом `--chunk-param-mode config`.

Запускаем контейнер (в нем примонтирована вся текущая папка):
```bash
bash start.sh
```

Внутри контейнера запускаем скрипт:
```bash
python3 da3_streaming.py --image_dir <путь к папке с картинками>
```

Чтобы не перезаписывать `chunk_size`, `overlap` и `loop_chunk_size` из конфига:
```bash
python3 da3_streaming.py --image_dir <путь к папке с картинками> --chunk-param-mode config
```

При запуски скрипта, возможно, начнут скачиваться недостающие веса, которые затем сохранятся в кэше и при следующем запуске загружаться не будут. На некоторых серверах из-за проблем с сетью веса не скачиваются, в таком случае можно загрузить вручную папку `.cache` в папку `da3_streaming`, эту папку можно найти на cds2 по пути `/home/lazarev_aa/da3_weights/.cache`. (для cds2 и aicenter неактуально)

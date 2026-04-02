# Quick Start

На вход трекеру нужны:
- предикты и эмбеддинги SAM3
- глубины, интринсики и экстринсики от DA3

Для `data/0` структура уже соответствует ожидаемой:
- `data/0/images`
- `data/0/sam3_outputs/tracks`
- `data/0/sam3_outputs/embeds`
- `data/0/da3_outputs`

Запуск трекера:

```bash
python static/run_tracker.py \
  data/0/images \
  data/0/sam3_outputs/tracks \
  data/0/sam3_outputs/embeds \
  data/0/da3_outputs \
  data/0/tracker_outputs
```

После запуска результаты сохраняются в:
- `data/0/tracker_outputs/outputs`
- `data/0/tracker_outputs/track_outputs`
- `data/0/tracker_outputs/meta_outputs`
- `data/0/tracker_outputs/filtered_outputs`
- `data/0/tracker_outputs/point_outputs`
- `data/0/tracker_outputs/rerun_export`

Сборка `.rrd` для Rerun:

```bash
python visualization/tracker_layers_rerun.py \
  --export-dir data/0/tracker_outputs \
  --save data/0/tracker_outputs/tracker_layers.rrd
```

# Что Где

`run_tracker.py` запускает полный пайплайн трекинга без ноутбука.

`tracker.py` содержит `Track`, `Simple3DTracker`, `VoxelMap` и правила обновления треков.

`utils/data.py` отвечает за загрузку данных, текстовые эмбеддинги и сохранение результатов.

`utils/mask.py` содержит эрозию масок и `merge_masks`, который разделяет слепленные объекты через DBSCAN.

`utils/point.py` строит 3D точки из DA3 depth/intrinsics/extrinsics и фильтрует их.

`utils/track_vis.py` делает 2D визуализацию треков поверх кадров.

`visualization/tracker_layers_rerun.py` строит `.rrd` из сохраненного `rerun_export`.

# Как Работает Трекер

Для каждого трека сохраняются:
- класс объекта
- маски по кадрам
- voxel-геометрия по кадрам

Основная метрика сопоставления:
- `w_ioa * ioa + w_dist * dist_similarity + w_emb * emb_similarity`

То есть используются:
- пересечение в 3D
- расстояние между центрами
- косинусная близость эмбеддингов

Порядок обработки:
- active SAM predict
- active best track
- lost best track
- tentative SAM predict
- tentative best track

Если совпадение не найдено, создается новый tentative track.

# Notes

Старый путь через `3d_tracker.ipynb` все еще полезен для отладки, но основной запуск теперь через `run_tracker.py`.

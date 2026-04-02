# Quick Start

На вход трекеру нужны:
- предикты и эмбеддинги SAM3
- глубины, интринсики и экстринсики от DA3

Ожидаемая структура входных данных:
- `<image_dir>`
- `<sam3_tracks_dir>`
- `<sam3_embeds_dir>`
- `<da3_outputs_dir>`

Запуск трекера:

```bash
python static/run_tracker.py \
  <image_dir> \
  <sam3_tracks_dir> \
  <sam3_embeds_dir> \
  <da3_outputs_dir> \
  <save_path>
```

После запуска результаты сохраняются в:
- `<save_path>/outputs`
- `<save_path>/track_outputs`
- `<save_path>/meta_outputs`
- `<save_path>/filtered_outputs`
- `<save_path>/point_outputs`
- `<save_path>/rerun_export`

Сборка `.rrd` для Rerun:

```bash
python visualization/tracker_layers_rerun.py \
  --export-dir <save_path> \
  --save <save_path>/tracker_layers.rrd
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

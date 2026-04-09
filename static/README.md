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
- `<save_path>/frame_000000.npz`
- `<save_path>/frame_000001.npz`
- ...

Каждый `frame_XXXXXX.npz` содержит:
- `frame_id`
- `image`
- `masks: dict[str, dict[int, np.ndarray]]`
- `embeddings: dict[str, dict[int, np.ndarray]]`
- `point_cloud`
- `intrinsic`
- `extrinsic`

Важно:
- id объектов в `masks` и `embeddings` это финальные `track.id` из трекера
- экспорт теперь один и тот же для визуализации и для `frame_selectors`

Сборка `.rrd` для Rerun:

```bash
python visualization/tracker_layers_rerun.py \
  --export-dir <save_path> \
  --save <save_path>/tracker_layers.rrd
```

# Что Где

`run_tracker.py` запускает полный пайплайн трекинга без ноутбука.

`tracker.py` содержит `Track`, `Simple3DTracker`, `VoxelMap` и правила обновления треков.

`utils/data.py` отвечает за загрузку данных, текстовые эмбеддинги и сохранение per-frame экспорта.

`utils/mask.py` содержит эрозию масок и `merge_masks`, который разделяет слепленные объекты через DBSCAN.

`utils/point.py` строит 3D точки из DA3 depth/intrinsics/extrinsics и фильтрует их.

`utils/track_vis.py` делает 2D визуализацию треков поверх кадров.

`visualization/tracker_layers_rerun.py` строит `.rrd` из `frame_*.npz` и на лету восстанавливает накопленную геометрию треков из `point_cloud + masks`.

`frame_selectors/base.py` читает тот же per-frame экспорт для выбора ключевых кадров.

# Как Работает Трекер

Во время трекинга для каждого трека поддерживаются:
- класс объекта
- маски по кадрам
- визуальные эмбеддинги по кадрам
- накопленная voxel-геометрия

В итоговый экспорт сохраняется не состояние трека целиком, а данные по каждому кадру:
- изображение
- видимые маски финальных треков
- эмбеддинги видимых треков
- point cloud сцены
- intrinsic
- extrinsic

Накопленная 3D геометрия для визуализации не сохраняется отдельно. `tracker_layers_rerun.py` восстанавливает ее онлайн из последовательности кадров.

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

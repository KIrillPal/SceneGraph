# Quick start

Собираем докер образ (на cds2 этот образ уже собран, поэтому сразу переходим к следующему шагу):

```bash
bash build.sh
```
Запускаем контейнер (в нем монтируется вся текущая папка с sam3):
```bash
bash start.sh
```
Внутри контейнера запускаем скрипт сегментации:
```bash
python3 run_inference.py <image_folder_path> <txt_with_objects_path> <save_path>
```
В скрипт нужно передать 3 парамертра: путь к папке с картинками, .txt файл с уникальными объектами на сцене (каждый объект с новой строчки), путь куда сохранить эмбеддинги и треки от sam3

# TODO

change current code version to SAM 3.1 release: https://github.com/facebookresearch/sam3/pull/503
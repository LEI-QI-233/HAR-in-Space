# Input format
for training the MicroG-4M on our modified PySlowFast codes by convert our dataset to an ava-like format

## Structure

```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|_ annotations
```


`frame_lists` folder is [here](../../../training/csv_files/frame_lists/)

`annotations` folder is [here](../../../training/csv_files/annotations/)

`frames` folder:  please use `runconverter` in [video2ava](../../convert2ava/video2ava/) to transfer the videos to pictures as shown above.

for downloading videos, please see [DATASET.md](../../../dataset/DATASET.md)

for training details, please see [TRAINING.md](../../../training/TRAINING.md)
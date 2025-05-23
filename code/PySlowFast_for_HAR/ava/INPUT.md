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

The entire ava folder and configs yaml file be download on our [MicroG-HAR-train-ready](https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready) Hugging Face repository.

Please check the details from [DATASET.md](../SlowFast/slowfast/datasets/DATASET.md) of PySlowFast.
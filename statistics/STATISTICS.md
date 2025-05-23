# Statistics and Evaluation

## Metadata of MicroG-4M
<iframe
  src="https://huggingface.co/datasets/LEI-QI-233/MicroG-4M/embed/viewer/actions/all"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## Metadata of [converted MicroG-4M](https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready)

to ava-like format, for fine-tuning via PyslowFast program. Only for HAR task.

<iframe
  src="https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready/embed/viewer/annotations/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## [statistics](./) Folder
This folder contains all statistical and evaluation information for the MicroG-4M dataset and its benchmark.

### [dataset_statistics](./dataset_statistics/) Folder
In this folder are all the statistics for the dataset itself and HAR task.

[HAR_Dataset_Statistics.pdf](statistics/HAR_Dataset_Statistics.pdf) organizes all the statistical data from raw csv files for easy viewing.

The csv files in the folder contain the raw information of different statistical data.

csv Files ending with 
- "**all**" represent statistics based on all data.
- "**movie**" represent statistics based on movies. 
- "**real**" represent statistics based on real videos.

### [benchmark_statistics](./benchmark_statistics/) Folder

In this folder are all the statistics and evaluation information for benchmark.

[HAR_finetune_eval.pdf](./benchmark_statistics/HAR_finetune_eval.pdf) organizes all the statistical data from raw csv files for easy viewing.
# Benchmarks

This document presents detailed benchmark information for MicroG-4M.

## Dataset files for fine-tuning and evaluation

In order to use the PySlowFast for training and evaluation HAR tass, we converted our dataset into a format almost compatible with the AVA dataset, sothat we can use [PySlowFast_for_HAR](../code/PySlowFast_for_HAR/) to train and evaluate.

The converted dataset files are hosted in [MicroG-HAR-train-ready](https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready) dataset repository on Hugging Face. where you can view metadata and download the full dataset. For detailed information and specifications, please refer to this [README](https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready/blob/main/README.md) file.

The [MicroG-HAR-train-ready](https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready) repository provides a convenient and fast way to train and evaluate our dataset and model using the [PySlowFast_for_HAR](../code/PySlowFast_for_HAR/) code. Alternatively, you can use the original dataset and convert it into any format you need.

## Fine-tuned Models
We host our fine-tuned models on [MicroG-4M-models](https://huggingface.co/LEI-QI-233/MicroG-4M-models) models repository on Hugging Face. You can download our fine-tuned models if you need. The information of models please refer [Benchmark Data](#har-task-performance-comparison-of-models-fine-tuned-on-microg-4m) below.

For the configuration parameters, please check [here](https://huggingface.co/datasets/LEI-QI-233/MicroG-HAR-train-ready/tree/main/configs/HAR).

For the baselines we used, please refer to [Model ZOO](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md) of PySlowFast.


## Benchmark Data

### HAR Task: Performance comparison of models fine-tuned on MicroG-4M

| Arch     | TC   | Backbone | #Params (M) | mAP (%) | F1-score (%) | Recall (%) | AUROC (%) |
| -------- | ---- | -------- | ----------- | ------- | ------------ | ---------- | --------- |
| C2D      | 8×8  | R50      | 23.61       | 29.51   | 8.09         | 6.58       | 83.49     |
| C2D NLN  | 8×8  | R50      | 30.97       | 44.64   | 28.30        | 24.86      | 89.40     |
| I3D      | 8×8  | R50      | 27.33       | 46.41   | 26.37        | 22.25      | 88.79     |
| I3D NLN  | 8×8  | R50      | 34.68       | 47.12   | 28.07        | 24.65      | 88.52     |
| Slow     | 8×8  | R50      | 31.74       | 45.19   | 26.13        | 22.77      | 88.49     |
| Slow     | 4×16 | R50      | 31.74       | 46.37   | 28.72        | 25.38      | 88.30     |
| SlowFast | 8×8  | R50      | 33.76       | 43.02   | 22.63        | 18.98      | 88.51     |
| SlowFast | 4×16 | R50      | 33.76       | 42.10   | 23.69        | 20.18      | 87.54     |
| MViTv1   | 16×4 | B-CONV   | 36.34       | 12.86   | 5.54         | 4.66       | 74.63     |
| MViTv2   | 16×4 | S        | 34.27       | 15.14   | 8.16         | 7.17       | 78.61     |
| X3D      | 13×6 | S        | 2.02        | 14.07   | 5.77         | 4.52       | 78.23     |
| X3D      | 16×5 | L        | 4.37        | 18.70   | 9.15         | 7.47       | 78.27     |


**Note:** 
- All models has been pretrained on Kinetics400 dataset and continually trained on MicroG-4M. 
- `TC` denotes the temporal configuration (frame length × sampling rate). 
- `#Params` indicates the number of parameters (in millions, M).

#### As a comparison:: Zero-shot performance on MicroG-4M test set for models pretrained on Kinetics and fine-tuned on AVA

| Arch     | TC   | Backbone | Pretraining  | Fine-tuning | mAP (%) | F1-score (%) | Recall (%) | AUROC (%) |
| -------- | ---- | -------- | ------------ | ----------- | ------- | ------------ | ---------- | --------- |
| Slow     | 8×8  | R50      | Kinetics-400 | AVA v2.2    | 16.24   | 2.67         | 1.99       | 73.83     |
| SlowFast | 32×2 | R101     | Kinetics-600 | AVA v2.2    | 23.81   | 6.32         | 6.62       | 77.83     |


**Note:**
- All metrics are macro-averaged over action classes. 
- mAP is measured at IoU = 0.5. F1 and AUROC are computed per class and then averaged. 
- `TC` denotes the temporal configuration (frame length × sampling rate).


### Video Captioning Task: Performance comparison across open-source and closed-source models on the MicroG-4M benchmark.

#### Open-Sourced Models:

| Model         | #f | Cider | Bleu-4 | Rouge-L | Meteor | S-BERT | BERTScore |
| ------------- | -- | ----- | ------ | ------- | ------ | ------ | --------- |
| Video-ChatGPT | 3  | 0.06  | 0.12   | 10.10   | 4.33   | 39.61  | 85.40     |
| mPLUG-Owl3    | 3  | 0.16  | 0.40   | 11.87   | 5.88   | 47.45  | 85.91     |
| LLaVA-NeXT    | 8  | 0.30  | 1.88   | 16.32   | 14.45  | 54.16  | 84.98     |
| Video-LLaVA   | 8  | 0.03  | 0.07   | 9.29    | 4.12   | 42.97  | 84.89     |
| Qwen2.5-VL    | 9  | 0.03  | 1.34   | 13.75   | 15.67  | 56.46  | 84.01     |
| InternVideo   | 90 | 0.77  | 2.60   | 16.57   | 15.18  | 55.28  | 85.41     |

#### Closed-Sourced Models:

| Model          | #f | Cider | Bleu-4 | Rouge-L | Meteor | S-BERT | BERTScore |
| -------------- | -- | ----- | ------ | ------- | ------ | ------ | --------- |
| GPT-4o         | 6  | 1.74  | 2.65   | 16.46   | 11.27  | 62.18  | 86.75     |
| Gemini 1.5 Pro | 16 | 3.52  | 3.28   | 17.34   | 15.19  | 63.38  | 86.25     |

**NOTE:**

- `#f` denotes the number of input frames used during model inference.


### Video Question Answering (VQA) Task: Performance comparison of VQA models on the MicroG-4M benchmark.

#### Open-Sourced Models:

| Model       | #f | Cider | Bleu-4 | Rouge-L | Meteor | S-VQA | BERTScore |
| ----------- | -- | ----- | ------ | ------- | ------ | ----- | --------- |
| LLaVA-NeXT  | 8  | 24.00 | 22.14  | 15.56   | 12.40  | 38.08 | 87.15     |
| Video-LLaVA | 8  | 25.70 | 28.47  | 15.90   | 10.71  | 35.39 | 87.13     |
| Qwen2.5-VL  | 9  | 2.99  | 0.65   | 8.35    | 8.47   | 40.65 | 84.80     |

#### Closed-Sourced Models:

| Model          | #f | Cider | Bleu-4 | Rouge-L | Meteor | S-VQA | BERTScore |
| -------------- | -- | ----- | ------ | ------- | ------ | ----- | --------- |
| Gemini 1.5 Pro | 16 | 8.78  | 1.33   | 13.03   | 12.54  | 43.15 | 86.41     |
| GPT-4o         | 6  | 33.98 | 3.76   | 18.11   | 15.89  | 44.56 | 87.81     |

**NOTE:**

- `#f` denotes the number of input frames used during model inference.

## Further Statistics and Evaluation Data

More statistics and evaluation infomation are available in [benchmark_statistics](../statistics/benchmark_statistics/) folder.

Please refer to [STATISTICS.md](../statistics/STATISTICS.md).


## [evaluation_results](./evaluation_results/)

This folder contains the raw data of each validated epoch and tested epoch generated by running the PySlowFast program for fine-tuning MicroG-4M. It includes:

- Log file of running the PySlowFast Program
- Evaluation metrics
- Per-class AP
- Detections recults

Check [benchmark_statistics](../statistics/benchmark_statistics/) folder for summarized data from CSVs here.

---
## Input Format for using the PyslowFast
For training input details, please check [INPUT.md](../code/PySlowFast_for_HAR/ava/INPUT.md)
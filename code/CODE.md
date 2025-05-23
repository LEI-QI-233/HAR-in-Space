# Related Code of MicroG-4M

The [code](./) folder contains all relevant code for the MicroG-4M dataset. Below is a detailed introduction to each subfolder.

## [create_dataset](create_dataset)

The code in this folder includes all the code we created this dataset and for HAR task.

## [convert2ava](convert2ava)

In order to use the PySlowFast library for training and evaluation, we converted our dataset into a format almost compatible with the AVA dataset, sothat we can use [PySlowFast_for_HAR](PySlowFast_for_HAR) to train and evaluate. This folder contains the code required for the conversion.

- [csv2ava](convert2ava/csv2ava/) converts a CSV file into an AVA-style CSV. The only difference is that we replace the original AVA dataset’s `key-timestamp` header with our `key-frame-stamp`.

- [video2ava](convert2ava/video2ava/) conterts the videos to pictures to suit the input format of [PySlowFast_for_HAR](PySlowFast_for_HAR).


## [PySlowFast_for_HAR](PySlowFast_for_HAR)

- This folder contains the modified PySlowFast code.

- We converted our dataset into an AVA-like format to match the program’s input requirements.

- We also modified the PySlowFast code so that, while using AVA-style inputs, it accepts our frame-stamp format.

- The core change replaces timestamp-in-seconds inputs with frame stamps.

- Information of Pyslowfast, such as installation and input format, please check [README](./PySlowFast_for_HAR/SlowFast/README.md) file of PySlowFast itself.

## [statistic_code](statistic_code)

This folder contains the code for generating statistics of MicroG-4M.

## [vqa_caption_eval](./vqa_caption_eval/) 
This folder contains all code for evaluating VQA and Captioning tasks.
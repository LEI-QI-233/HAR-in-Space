# Instruction of codes folder

## [PySlowFast_for_HAR](PySlowFast_for_HAR)

This folder is contains the code for training, validating and testing the model through a modified PySlowFast code.

We convert out dataset to an ava-like format to adjust the input of this program.

our configuration file is in [here](./PySlowFast_for_HAR/SlowFast/configs/AVA/HAR/)

use [video2ava](convert2ava/video2ava/) to generate frame folder, and copy folders from [csv_files](../training/csv_files/), put them in [ava](./PySlowFast_for_HAR/ava/)folder.then you can run the pyslowfast program with those input and [configs](./PySlowFast_for_HAR/SlowFast/configs/AVA/HAR/).

detail of Pyslowfast please check [here](./PySlowFast_for_HAR/SlowFast/README.md)

## [convert2ava](convert2ava)

Codes in this folder is to convert the original dataset to a ava-like formast to train and test on [PySlowFast_for_HAR](PySlowFast_for_HAR)


[csv2ava](convert2ava/csv2ava/) conterts the csv file to ava-like csv file. The different is we change key-time-stamp to key-frame-stamp

the csv files are already converted and in [csv_files](../training/csv_files/). You can also find how we divide train val test set to train the model.

[video2ava](convert2ava/video2ava/) conterts the videos to pictures to suit the input format of [PySlowFast_for_HAR](PySlowFast_for_HAR).

## [create_dataset](create_dataset)

it contains the codes of creating this dataset.


## [statistic_code](statistic_code)

it contains the code to do the statistics in [statistics](../dataset/statistics/) folder.

## [vqa_caption_eval](./vqa_caption_eval/) 
is the folder of all codes for evaluating VQA and Captioning
# training folder

## [evaluation_results](./%20evaluation_results/)

here are the raw data of training validation and test on PySlowFast.


## [configs_backup](./configs_backup/)

are same configuration files in [here](../codes/PySlowFast_for_HAR/SlowFast/configs/AVA/HAR/) as a backup.


## [csv_files](./csv_files/) 

are the csv files from [original dataset](../dataset/csv_files/) convert to ava-like format and seperate to three partition: train val test, to be the input of PySlowFast.

the different from ava format is that we use key frame stamp instead of key time stamp. csv file records frame id directly.

you can check [dataset.md](../dataset/DATASET.md) to know how to transfer frame id to original second.


for training input details, please check [INPUT.md](../codes/PySlowFast_for_HAR/ava/INPUT.md)
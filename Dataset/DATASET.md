# Dataset Indroduction
Here are the details about this dataset. Including the description and format of each file. And the video description and download.

## Videos

### Download
Please download the zip file of all videos from [here](https://drive.google.com/file/d/1E8WBvKzWOEYbKwQw7qGeoN8fI0OrwQDl/view?usp=sharing)

### Folder Structure
The video folder has the following structure:

```
HAR_IN_SPACE_VIDEOS
|_ videos
|  |_ movie
|  |  |_ [movie name 0]
|  |  |  |_ [movie name 0]_001.mp4
|  |  |  |_ [movie name 0]_003.mp4
|  |  |  |_ ...
|  |  |_ [movie name 1]
|  |  |  |_ ...
|  |_ real
|  |  |_ [real video id 0]
|  |  |  |_[real video id 0]_002.mp4
|  |  |  |_[real video id 0]_003.mp4
|  |  |  |_ ...
|  |  |_ [real video id 1]
|  |  |  |_ ...
|_ videos_annotated
|  |_ [same structure as "videos" folder]
```
### Descriptions
All videos are **three-second, 480p, 30fps, H.264** video clips.  

In the `movie` folder, `movie name` is the name of the movie. 

In the `real` folder, all videos are downloaded from YouTube. `real video id` is the video id on YouTube. 

All video clips are valid clips and completely correspond to the [csv files](csv_files).

All video clips have been screened, irrelevant and invalid video clips have been deleted, such as scenes that are not in space in the movie and scenes where no one appears in the actual video. Therefore, the video clip serial numbers in the video folder are **not continuous**. 

The video serial number can be used to infer the position of the video clip in the complete video. The method is to multiply the serial number by 3 to equal the second of the original video.

## Folders and Files

### [csv_files](csv_files/)
This folder contains all the csv files of the dataset.
#### [actions.csv](csv_files/actions.csv)
The content of this file is the action sequence number of the characters framed by the bounding box in all videos.  
The headers are:
- `video_id`
- `movie_or_real`: m for movie and r for real
- `person_id`
- `action`: the id of action. Details see [label_map](Dataset/label_map/label_map.pbtxt)

#### [bounding_boxes.csv](csv_files/bounding_boxes.csv)
The content of this file is the bounding box information of all videos, which corresponds to the [actions.csv](Dataset/csv_files/actions.csv) with video_id and person_id.  
The headers are:
- `video_id`
- `frame_id`: Not a continuous number. Only frames where the bounding box is drawn. If no person is detected, ignore this frame.
- `person_id`
- `xmin`: x coordinate of the upper left corner (in pixels)

- `ymin`: y coordinate of the upper left corner

- `xmax`: x coordinate of the lower right corner

- `ymax`: y coordinate of the lower right corner

**Note**: All bounding boxes use pixel coordinates. Normalized coordinates are not used in this dataset, and all values ​​are pixel coordinates at the original resolution of the image.


### [label_map](label_map/)
This folder contains the label map
#### [label_map.pbtxt](label_map/label_map.pbtxt)
This label map is modified based on the label map of the [ava dataset](https://research.google.com/ava/index.html).The format is consistent with [ava_action_list_v2.2.pbtxt](https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt)

### [statistics](statistics/)
In this folder are all the statistics for the dataset.

[overall_result.csv](statistics/overall_result.csv) is a statistical summary of the overall quantity, such as the total number of videos, the number of action tags, and the total number of action types.

The other files contain detailed information about these statistics, such as the number of each action and the percentage of the total.

Files ending with 
- "**all**" represent statistics based on all data.
- "**movie**" represent statistics based on movies. 
- "**real**" represent statistics based on real videos.
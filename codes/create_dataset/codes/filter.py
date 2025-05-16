import os
from ultralytics import YOLO
import torch
import logging
import sys
import time
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

def is_video_file(filename):
    """Check if a file is a video based on its extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def get_all_video_paths(input_dir):
    """Recursively retrieve all video file paths from the input directory."""
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if is_video_file(file):
                video_paths.append(os.path.join(root, file))
    return video_paths

def process_video(model, video_path, confidence_threshold=0.6):
    """
    Process a single video to calculate the percentage of frames
    where a person is detected with confidence greater than the threshold.

    Args:
        model: The YOLO model.
        video_path: Path to the video file.
        confidence_threshold: Minimum confidence required to consider a detection valid.

    Returns:
       tuple: (video_name, person_frame_percentage, scene_changes)
    """
    total_frames = 0
    person_frames = 0

    # Start tracking with verbose=False to suppress detailed logs
    results_generator = model.track(
        source=video_path,
        stream=True,
        verbose=False
    )

    for result in results_generator:
        total_frames += 1
        # Check if any detected object is a person (class ID 0) with confidence > threshold
        persons = [obj for obj in result.boxes if obj.cls == 0 and obj.conf > confidence_threshold]
        if len(persons) > 0:
            person_frames += 1

    if total_frames == 0:
        percentage = 0.0
    else:
        percentage = (person_frames / total_frames) * 100
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Detect scene changes
    scene_changes = detect_scene_changes(video_path)
    if scene_changes is None:
        print(f"Scene detection failed for video: {video_name}")
    
    return (video_name, percentage, scene_changes)

def detect_scene_changes(video_path, threshold=40.0):
    """
    Detect the number of scene changes in a video.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold for scene detection sensitivity.

    Returns:
        int: Number of scene changes detected.
    """
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    try:
        video = open_video(video_path)
    except Exception as e:
        print(f"Failed to open video for scene detection: {video_path}. Error: {e}")
        return None

    try:
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        # Number of scene changes is one less than the number of scenes
        scene_changes = max(len(scene_list) - 1, 0)
        return scene_changes
    except Exception as e:
        print(f"Scene detection failed for video: {video_path}. Error: {e}")
        return None

def setup_logging(log_filename="../logs/filter.log"):
    """Set up logging to file and stdout."""
    if __name__ == "__main__":
        # Create log directory if it doesn't exist
        os.makedirs("../logs/", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='w'),  # Write log to file
                logging.StreamHandler(sys.stdout)   # Output to console
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)   # Output to console
            ]
        )
    # Redirect print function to logging
    global print
    print = logging.info

def delete_video_if_needed(video_path, percentage, scene_changes, percentage_threshold=60.0, scene_changes_threshold=0):
    """
    Delete the video file if the percentage of person frames is below the threshold
    or the number of scene changes exceeds the threshold.

    Args:
        video_path (str): Path to the video file.
        percentage (float): Percentage of frames with person detected.
        scene_changes (int): Number of scene changes detected.
        percentage_threshold (float, optional): Threshold for person frame percentage. Defaults to 60.0.
        scene_changes_threshold (int, optional): Threshold for number of scene changes. Defaults to 1.
    Returns:
        bool: True if the video is deleted. False if video is not deleted.
    """
    if percentage < percentage_threshold or scene_changes > scene_changes_threshold:
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            os.remove(video_path)
            print(f"Deleted video: '{video_name}' due to percentage = {percentage:.2f} < {percentage_threshold}% or scene_changes = {scene_changes} > {scene_changes_threshold}.\n")
            return True
        except Exception as e:
            print(f"Failed to delete video: {video_path}. Error: {e}")
            return False
    else:
        return False
    
def remove_empty_dirs(input_dir):
    """
    Remove all empty subdirectories in the specified directory.

    Args:
        input_dir (str): Path to the directory where empty subdirectories should be removed.
    """
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")

def run(logfile=False):
    """
    Main function to process all videos in the input directory and print results.
    """
    
    # Input directory
    INPUT_DIR = "../processed_videos/"

    # Path of YOLO model
    MODEL_PATH = "../yolo model/yolo11n.pt"
    
    if logfile:
        setup_logging()
        
    print("Start Program: filter ...")
    start_time = time.time()

    # Check CUDA availability
    if torch.cuda.is_available():
        print("Using CUDA")
        model = YOLO(MODEL_PATH).to('cuda')  # Use GPU
    else:
        print("CUDA unavailable, Using CPU")
        model = YOLO(MODEL_PATH)  # Use CPU

    # Get all video paths
    print("Scanning for video files...")
    video_paths = get_all_video_paths(INPUT_DIR)

    if not video_paths:
        print("No video files found in the specified directory.")
        return

    print(f"Found {len(video_paths)} video(s). Processing...\n")
    
    processed_videos = 0  # Initialize processed video counter

    for video_path in video_paths:
        video_result = process_video(model, video_path)
        video, percentage, scene_changes = video_result
        print(f"({processed_videos + 1}/{len(video_paths)})Processing video: '{video}'...")
        if percentage is not None and scene_changes is not None:
            if delete_video_if_needed(video_path, percentage, scene_changes):
                processed_videos += 1  # Increment counter after processing each video
                continue
            print(f"Person Frame Percentage: {percentage:.2f}%\nScene Changes: {scene_changes}\n")
        else:
            print(f"Person Frame Percentage: Error processing video.\nScene Changes: Unable to detect.\n")
        processed_videos += 1  # Increment counter after processing each video
    
    remove_empty_dirs(INPUT_DIR)
    elapsed_time = time.time() - start_time
    
    if elapsed_time < 60:
        time_display = f"{elapsed_time:.2f} seconds"
    else:
        minutes, seconds = divmod(elapsed_time, 60)
        time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"
    
    # Add the number of processed videos to the final output
    print(f"Filter finished. Total time: {time_display}. Number of processed videos: {processed_videos}.")
    print("")

if __name__ == "__main__":
    run(False)

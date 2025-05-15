import os
import glob
import shutil
import time
import ffmpeg
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

def clear_output_directory(output_dir):
    """
    Clear the output directory by removing its contents and recreating it.
    
    Args:
        output_dir (str): Path to the output directory.
    """
    if os.path.exists(output_dir):
        print("Output directory not empty, clearing...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def get_video_duration(video_path):
    """
    Get the duration of a video file using ffprobe.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        float: Duration of the video in seconds. Returns 0 if an error occurs.
    """
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0

def get_video_fps(video_path):
    """
    Get the frame rate (FPS) of a video file using ffprobe.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        float: Frame rate of the video. Defaults to 30.0 FPS if unable to retrieve.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            fps_fraction = video_stream['r_frame_rate'].split('/')
            if len(fps_fraction) == 2:
                fps = int(fps_fraction[0]) / int(fps_fraction[1])
            else:
                fps = float(fps_fraction[0])
            return fps
    except ffmpeg.Error as e:
        print(f"Error getting FPS for {video_path}: {e}")
        return 30.0  # Default to 30 FPS if unable to retrieve

def split_video_into_segments(video_path, output_dir):
    """
    Split the video into 3-second segments using ffmpeg, forcing keyframes at exact intervals.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the output directory where segments will be saved.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing '{video_name}'...")
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    fps = get_video_fps(video_path)
    segment_time = 3  # Desired segment duration in seconds

    # Calculate total duration and generate keyframe expression
    duration = get_video_duration(video_path)
    if duration == 0:
        print(f"Skipping video '{video_name}' due to zero duration.")
        return

    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                f"{video_output_dir}/{video_name}_%03d.mp4",
                f='segment',
                reset_timestamps='1',
                segment_time=segment_time,
                force_key_frames=f"expr:gte(t,n_forced*{segment_time})",
                vcodec='libx264',
                r=30,
                an=None
            )
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print(f"An error occurred while processing video '{video_name}': {e}")

def remove_short_final_segments(video_output_dir, video_name):
    """
    Remove the final segment if its duration is less than 3 seconds.
    
    Args:
        video_output_dir (str): Path to the directory containing video segments.
        video_name (str): Name of the original video file without extension.
    """
    segment_files = sorted(glob.glob(f"{video_output_dir}/{video_name}_*.mp4"))
    if segment_files:
        last_segment = segment_files[-1]
        duration = get_video_duration(last_segment)
        if duration < 3:
            os.remove(last_segment)

def is_valid_video(video_path):
    """
    Check if the file is a valid video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        bool: True if the file contains a video stream, False otherwise.
    """
    try:
        probe = ffmpeg.probe(video_path)
        return any(stream['codec_type'] == 'video' for stream in probe['streams'])
    except ffmpeg.Error:
        return False

def process_video(video_path, input_dir, output_dir):
    """
    Process a single video: split into segments and remove short final segment.
    
    Args:
        video_path (str): Path to the input video file.
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
    """
    if is_valid_video(video_path):
        # Get the relative path to maintain folder structure
        relative_path = os.path.relpath(video_path, input_dir)
        relative_dir = os.path.dirname(relative_path)
        # Define the corresponding output directory
        current_output_dir = os.path.join(output_dir, relative_dir)
        split_video_into_segments(video_path, current_output_dir)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(current_output_dir, video_name)
        remove_short_final_segments(video_output_dir, video_name)
        print(f"'{video_name}' is finished")
    else:
        print(f"Invalid video file skipped: {video_path}")

def split_Video(input_dir, output_dir):
    """
    Split all videos in the input directory and its subdirectories into segments concurrently.
    
    Args:
        input_dir (str): Path to the directory containing original videos.
        output_dir (str): Path to the directory where segmented videos will be saved.
    """
    clear_output_directory(output_dir)
    # Use glob to recursively find all .mp4 files
    video_files = glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
    
    # Using ThreadPoolExecutor to handle files concurrently
    logical_cores = os.cpu_count()
    print(f"#Logical cores = {logical_cores}")
    with ThreadPoolExecutor(max_workers=logical_cores) as executor:
        executor.map(lambda p: process_video(p, input_dir, output_dir), video_files)

def setup_logging(log_filename="../logs/splitVideo.log"):
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

def run(logfile = False,input_dir = "../original_videos/"):
    """
    Run the process to split all videos in the directory.
    
    Args:
        logfile(bool): If generate the log file. Default to False
        input_dir(str):The directory of input videos. Default to "../original_videos/"
    """
    output_dir = "../processed_videos/"
    
    if logfile:
        setup_logging()
    
    print("Start Program: Split Video...")
    start_time = time.time()
    split_Video(input_dir, output_dir)
    elapsed_time = time.time() - start_time
    
    if elapsed_time < 60:
        time_display = f"{elapsed_time:.2f} seconds"
    else:
        minutes, seconds = divmod(elapsed_time, 60)
        time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"
    
    print(f"Split video finished. Total time: {time_display}.")
    print("")

if __name__ == "__main__":
    run()
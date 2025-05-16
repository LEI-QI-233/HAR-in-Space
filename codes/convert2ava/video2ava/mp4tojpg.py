import os
import ffmpeg
import shutil
import logging
import sys
import time

def extract_all_frames_with_structure(input_root, output_root, fps=30):

    video_files = []
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_files.append(os.path.join(dirpath, filename))
    
    total_videos = len(video_files)
    processed_videos = 0

    for input_path in video_files:
        processed_videos += 1

        rel_path = os.path.relpath(input_path, input_root)
        rel_dir = os.path.dirname(rel_path)
        clip_name = os.path.splitext(os.path.basename(input_path))[0]

        out_dir = os.path.join(output_root, rel_dir, clip_name)
        os.makedirs(out_dir, exist_ok=True)

        out_pattern = os.path.join(out_dir, f'{clip_name}_%06d.jpg')

        print(f"[{processed_videos}/{total_videos}] processing {clip_name}")

        try:
            (
                ffmpeg
                .input(input_path)
                .output(out_pattern, r=fps, qscale=1)
                .run(overwrite_output=True, quiet=True)
            )
            print("OK\n")
        except ffmpeg.Error as e:
            print(f"[ERROR] Failed on {clip_name}\n")
            print(e.stderr.decode())

def clear_output_folder(output_root):
    if os.path.exists(output_root):
        print("Clearing existing output folder\n")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    
def setup_logging(log_filename="../logs/mp4tojpg.log"):
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

def run(logfile = False):
    input = "../input"
    output = "../output"
    
    if logfile:
        setup_logging()
        
    print("Start Program: mp4tojpg ...\n")
    print(f"log file = {logfile}\n")
    start_time = time.time()
    
    clear_output_folder(output)
    extract_all_frames_with_structure(input, output)
    
    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        time_display = f"{elapsed_time:.2f} seconds"
    else:
        minutes, seconds = divmod(elapsed_time, 60)
        time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"
    
    # Add the number of processed videos to the final output
    print(f"\nProgram finished. Total time: {time_display}.")
    print("")

if __name__ == "__main__":
    run()

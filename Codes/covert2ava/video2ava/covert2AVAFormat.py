import os
import shutil
from pathlib import Path
import logging
import sys
import time

def flatten_to_ava_format(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    clip_dirs = list(src_root.rglob("*"))
    clip_dirs = [d for d in clip_dirs if d.is_dir() and any(f.suffix == ".jpg" for f in d.iterdir())]
    total_video_amount = len(clip_dirs)
    current_video = 1
    
    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        print(f"{current_video}/{total_video_amount} process {clip_name}")
        target_dir = dst_root / clip_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img in sorted(clip_dir.glob("*.jpg")):
            shutil.copy(img, target_dir / img.name)
        
        print(f"ok\n")
        current_video = current_video + 1
        
def clear_output_folder(output_root):
    if os.path.exists(output_root):
        print("Clearing existing output folder\n")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
            
            
def setup_logging(log_filename="../logs/covert2AVAFormat.log"):
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
    
    input = "../output"
    output = "../ava/frames"
    
    if logfile:
        setup_logging()
    
    print("Start Program: covert2AVAFormat ...\n")
    print(f"log file = {logfile}\n")
    start_time = time.time()
    
    clear_output_folder(output)
    flatten_to_ava_format(input,output)
    
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
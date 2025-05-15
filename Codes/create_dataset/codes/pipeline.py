import splitVideo
import filter
import trackPerson
import logging
import os
import sys
import time


def setup_logging(log_filename="../logs/pipeline.log"):
    # setup log file
    os.makedirs("../logs/",exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),  # write log into file
            logging.StreamHandler(sys.stdout)   # output to cmd
        ]
    )
    # redirect print-function
    global print
    print = logging.info
    
def run(logfile = True):
    """
    Run the pipeline
    
    Input directory:
        original_videos
    
    Output directory:
        processed_videos: for processed videos
        annonated_videos: for annonated videos
        logs: for the log file of this pipeline
        bounding_box: for all bounding box information
    """
    if logfile:
        setup_logging()
    
    start_time = time.time()
    print("Start pipeline...")
    print("")
    splitVideo.run(logfile = logfile)
    filter.run(logfile = logfile)
    trackPerson.run(logfile = logfile)
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time < 60:
        time_display = f"{elapsed_time:.2f} seconds"
    else:
        minutes, seconds = divmod(elapsed_time, 60)
        time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"

    print(f"Pipeline finished. Total time: {time_display}. Whole program ends.")

if __name__ == "__main__":
    run()
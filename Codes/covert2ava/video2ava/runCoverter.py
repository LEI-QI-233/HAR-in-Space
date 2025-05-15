import logging
import os
import sys
import time
import covert2AVAFormat
import mp4tojpg

def setup_logging(log_filename="../logs/runCoverter.log"):
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
    
    if logfile:
        setup_logging()
    
    start_time = time.time()
    print("Start coverter...")
    print("")
    mp4tojpg.run(logfile = logfile)
    covert2AVAFormat.run(logfile = logfile)
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time < 60:
        time_display = f"{elapsed_time:.2f} seconds"
    else:
        minutes, seconds = divmod(elapsed_time, 60)
        time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"

    print(f"Coverter finished. Total time: {time_display}. Whole program ends.")

if __name__ == "__main__":
    run()
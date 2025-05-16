import os
import subprocess
import signal
import time

def run_with_nohup():
    # Ensure ../docs and ../logs directories exist
    os.makedirs("../docs", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    
    # Check if there is already a running process
    if check_pid_status(print_message=False, show_log=False):  # Check status without printing log
        choice = input("A process is already running. Do you want to terminate it and start a new one? (y/n): ")
        if choice.lower() != 'y':
            print("Process start canceled.")
            return
        else:
            terminate_pid()
    
    # Command to start the process and redirect output to pipeline.log
    command = "nohup python3 main.py > ../logs/pipeline.log 2>&1 & echo $!"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Get and save the PID if it's valid
    pid = result.stdout.strip()
    if pid.isdigit():
        with open("../docs/main_pid.txt", "w") as pid_file:
            pid_file.write(pid)
        print(f"main.py has started with nohup, PID: {pid}, saved to ../docs/main_pid.txt")
    else:
        print("Failed to start, unable to retrieve process PID. Please check if main.py is running properly.")

def check_pid_status(print_message=True, show_log=True):
    log_file_path = "../logs/pipeline.log"
    try:
        with open("../docs/main_pid.txt", "r") as pid_file:
            pid = pid_file.read().strip()
        
        if pid and pid.isdigit():
            pid = int(pid)
            try:
                os.kill(pid, 0)  # Check if process is running
                if print_message:
                    print(f"Process with PID {pid} is currently running.\n")
                
                # Display the latest log block only if show_log is True
                if show_log:
                    try:
                        with open(log_file_path, "r") as log_file:
                            log_lines = log_file.readlines()
                            if log_lines:
                                # Find the last non-empty line and go up to the nearest empty line
                                last_log_block = []
                                for line in reversed(log_lines):
                                    if line.strip():  # Stop if we reach a non-empty line
                                        last_log_block.append(line)
                                    elif last_log_block:  # Stop collecting if we hit an empty line after finding log content
                                        break
                                
                                # Reverse to print in the correct order
                                last_log_block.reverse()
                                print("Latest log block:")
                                for line in last_log_block:
                                    print(line, end="")  # Print each line in the log block
                            else:
                                print("Log file is empty.")
                    except FileNotFoundError:
                        print("Log file not found. Make sure the process is generating output.")
                return True  # Process is running
            except ProcessLookupError:
                if print_message:
                    print(f"Process with PID {pid} does not exist, it may have finished or been terminated.")
                return False  # Process not found
        else:
            if print_message:
                print("No valid PID found in main_pid.txt. Please run main.py first.")
            return False  # No valid PID
    
    except FileNotFoundError:
        if print_message:
            print("PID file ../docs/main_pid.txt not found. Please start the program to save the PID.")
        return False  # PID file not found

def terminate_pid():
    try:
        with open("../docs/main_pid.txt", "r") as pid_file:
            pid = pid_file.read().strip()
        
        if pid and pid.isdigit():
            confirm = input(f"Are you sure you want to terminate the process with PID {pid}? (y/n): ")
            if confirm.lower() == 'y':
                pid = int(pid)
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Process with PID {pid} has been successfully terminated.")
                    os.remove("../docs/main_pid.txt")  # Remove PID file to avoid accidental reuse
                except ProcessLookupError:
                    print(f"Process with PID {pid} does not exist, it may have finished or been terminated.")
                except PermissionError:
                    print(f"Permission denied to terminate process with PID {pid}.")
            else:
                print("Process termination canceled.")
        else:
            print("No valid PID found in main_pid.txt.")
    
    except FileNotFoundError:
        print("PID file ../docs/main_pid.txt not found. Unable to terminate process.")

def main():
    while True:
        print("\nPlease select an option:")
        print("1. Start main.py and save PID")
        print("2. Check process status and display latest log output")
        print("3. Terminate process")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")
        
        if choice == "1":
            run_with_nohup()
        elif choice == "2":
            check_pid_status(show_log=True)  # Show log when checking status
        elif choice == "3":
            terminate_pid()
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()

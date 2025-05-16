import os
import subprocess
import signal
import tkinter as tk
from tkinter import font, filedialog
import re

class ProgramManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Program Manager")

        # Set the initial size of the screen, minimum size to current dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        initial_width = max(600, screen_width // 4)
        initial_height = max(420, screen_height // 4)
        self.root.geometry(f"{initial_width}x{initial_height}")
        self.root.minsize(480, 360)

        # Configure grid to be resizable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=3)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Set up a modern-looking font similar to Apple's style
        header_font = font.Font(family="Helvetica", size=15, weight="bold")
        button_font = font.Font(family="Helvetica", size=13, weight="bold")
        log_font = font.Font(family="Helvetica", size=11)

        # Create a label for the header
        header_label = tk.Label(root, text="Manage Background Program on the Server", font=header_font, fg="#000000", pady=5)
        header_label.grid(row=0, column=0, columnspan=2, pady=(10, 10), padx=10, sticky="nsew")

        # Set up the GUI layout with distinct colors for buttons
        start_button_style = {
            "font": button_font,
            "bg": "#4186DD",  # Deeper blue for Start Program button
            "fg": "#ffffff",  # White text for better contrast
            "activebackground": "#366EB8",  # Deeper shade when clicked
            "highlightthickness": 0,  # Remove border to keep buttons clean
            "bd": 0,
        }

        status_button_style = {
            "font": button_font,
            "bg": "#5AA469",  # Distinct green for Check Status button
            "fg": "#ffffff",  # White text for better contrast
            "activebackground": "#4A8B57",  # Deeper shade when clicked
            "highlightthickness": 0,
            "bd": 0,
        }

        terminate_button_style = {
            "font": button_font,
            "bg": "#E57373",  # Red for Terminate Process button
            "fg": "#ffffff",
            "activebackground": "#D64D4D",
            "highlightthickness": 0,
            "bd": 0,
        }

        exit_button_style = {
            "font": button_font,
            "bg": "#FFA726",  # Orange for Exit button
            "fg": "#ffffff",
            "activebackground": "#FB8C00",
            "highlightthickness": 0,
            "bd": 0,
        }

        # Add buttons with different colors to distinguish actions
        tk.Button(root, text="Start Program", command=self.run_with_nohup, **start_button_style).grid(row=1, column=0, padx=(10, 10), pady=10, sticky="nsew")
        tk.Button(root, text="Check Status", command=self.check_status_and_show_log, **status_button_style).grid(row=1, column=1, padx=(10, 10), pady=10, sticky="nsew")
        tk.Button(root, text="Terminate Program", command=self.terminate_process, **terminate_button_style).grid(row=2, column=0, padx=(10, 10), pady=10, sticky="nsew")
        tk.Button(root, text="Exit", command=self.exit_program, **exit_button_style).grid(row=2, column=1, padx=(10, 10), pady=10, sticky="nsew")
        
        # Frame to hold the log display and scrollbar
        log_frame = tk.Frame(root)
        log_frame.grid(row=3, column=0, columnspan=2, padx=(10, 10), pady=(10, 20), sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        # Text widget to display log output with improved contrast and color tagging
        self.log_display = tk.Text(log_frame, width=70, height=15, wrap=tk.WORD, font=log_font, bg="#F8F8F8", fg="#333333", borderwidth=1, relief="flat")
        self.log_display.grid(row=0, column=0, sticky="nsew")
        self.log_display.config(state=tk.DISABLED)

        # Scrollbar for the log display
        log_scrollbar = tk.Scrollbar(log_frame, command=self.log_display.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_display.config(yscrollcommand=log_scrollbar.set)

        # Keep track of the log file path
        self.log_file_path = None  # Initialize to None

    def run_with_nohup(self):
        # Ensure necessary directories exist
        os.makedirs("../pid_docs", exist_ok=True)
        os.makedirs("../logs", exist_ok=True)

        # Open a file dialog to select a Python file
        file_path = filedialog.askopenfilename(title="Select Python Program to Run", filetypes=[("Python Files", "*.py")])

        if not file_path:
            self.update_log_display("No file selected.\n")
            return

        # Get the base name of the selected script (e.g., 'pipeline.py')
        script_name = os.path.basename(file_path)
        # Remove the '.py' extension to use for the log file name
        log_name = os.path.splitext(script_name)[0] + ".log"
        # Set the log file path accordingly
        self.log_file_path = os.path.join("../logs", log_name)

        # Check if a process is already running
        if self.check_pid_status(show_log=False):
            if not self.custom_yesno_dialog("Program Running", "A program is already running.\n\nWould you like to terminate it and start a new one?"):
                self.update_log_display("Program start canceled.\n")
                return
            else:
                self.terminate_process()

        # Command to start the selected process
        command = f"nohup python3 {file_path} >/dev/null 2>&1 & echo $!"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Get and save the PID if it's valid
        pid = result.stdout.strip()
        if pid.isdigit():
            with open("../pid_docs/main_pid.txt", "w") as pid_file:
                pid_file.write(pid)
            with open("../pid_docs/script_name.txt", "w") as script_file:
                script_file.write(script_name)
            self.update_log_display(f"Program '{script_name}' has started with PID {pid}.\n")
        else:
            self.update_log_display("Failed to start the program.\n")

    def check_pid_status(self, show_log=True):
        output = ""
        try:
            with open("../pid_docs/main_pid.txt", "r") as pid_file:
                pid = pid_file.read().strip()
            
            if pid and pid.isdigit():
                pid = int(pid)
                try:
                    # os.kill(pid, 0)  # Check if process is running
                    # output += f"The program with PID {pid} is running.\n"

                    # Load the script name and set the log file path
                    try:
                        with open("../pid_docs/script_name.txt", "r") as script_file:
                            script_name = script_file.read().strip()
                        log_name = os.path.splitext(script_name)[0] + ".log"
                        self.log_file_path = os.path.join("../logs", log_name)
                    except FileNotFoundError:
                        output += "'script_name.txt' is not found. Unable to determine log file.\n"
                        if show_log:
                            self.update_log_display(output)
                        return True  # Process is running, but log file path is unknown
                    
                    os.kill(pid, 0)  # Check if process is running
                    output += f"The program '{script_name}' with PID {pid} is running.\n"

                    # Display the latest log block only if show_log is True
                    if show_log:
                        try:
                            with open(self.log_file_path, "r") as log_file:
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
                                    output += "\nLatest log block:\n" + "".join(last_log_block)
                                else:
                                    output += "Log file is empty.\n"
                        except FileNotFoundError:
                            output += "Log file is empty.\n"
                    if show_log:
                        self.update_log_display(output)
                    return True  # Process is running
                except ProcessLookupError:
                    output = f"Program '{script_name}' with PID {pid} is no longer running.\n"
                    if show_log:
                        self.update_log_display(output)
                    # Clean up pid_docs folder
                    self.clean_pid_docs_and_empty_logs()
                    return False  # Process not found
            else:
                output = "No program is running in the background.\n"
                if show_log:
                    self.update_log_display(output)
                # Clean up pid_docs folder
                self.clean_pid_docs_and_empty_logs()
                return False  # No valid PID

        except FileNotFoundError:
            output = "No program is running in the background.\n"
            if show_log:
                self.update_log_display(output)
            return False  # PID file not found

    def check_status_and_show_log(self):
        # Check process status and display the latest log block
        self.check_pid_status(show_log=True)
        
    def clean_directory(self, dir_path):
        # Check if the directory exists
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Get all files in the directory
            files_in_dir = os.listdir(dir_path)
            # Delete empty files
            for filename in files_in_dir:
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)
            # Check if the directory is empty after deletions
            files_in_dir = os.listdir(dir_path)
            if not files_in_dir:
                # Directory is empty, delete it
                os.rmdir(dir_path)

    def terminate_process(self):
        try:
            with open("../pid_docs/main_pid.txt", "r") as pid_file:
                pid = pid_file.read().strip()
            
            if pid and pid.isdigit():
                if not self.custom_yesno_dialog("Terminate Program", f"Are you sure you want to terminate the program with PID {pid}?"):
                    self.update_log_display("Program termination canceled.\n")
                    return
                
                pid = int(pid)
                try:
                    os.kill(pid, signal.SIGTERM)
                    self.update_log_display(f"Program with PID {pid} has been terminated.\n")
                    self.clean_pid_docs_and_empty_logs()  # Clean up pid_docs folder and logs folder if it is empty
                except ProcessLookupError:
                    self.update_log_display(f"Program with PID {pid} does not exist.\n")
                    # Clean up pid_docs folder
                    self.clean_pid_docs_and_empty_logs()
                except PermissionError:
                    self.update_log_display(f"Permission denied to terminate program with PID {pid}.\n")
            else:
                self.update_log_display("No valid PID found in main_pid.txt.\n")
                # Clean up pid_docs folder
                self.clean_pid_docs_and_empty_logs()
        
        except FileNotFoundError:
            self.update_log_display("No program is running in the background.\n")
            
    def exit_program(self):
        # Check if process is running
        if not self.check_pid_status(show_log=False):
            # Clean up pid_docs folder
            self.clean_pid_docs_and_empty_logs()
        # Close the application
        self.root.quit()

    def custom_yesno_dialog(self, title, message):
        """Create a custom yes/no dialog box to replace messagebox."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.grab_set()  # Make sure the dialog is modal

        # Set up fonts
        message_font = font.Font(family="Helvetica", size=12)
        button_font = font.Font(family="Helvetica", size=11, weight="bold")

        # Set up the dialog layout
        message_label = tk.Label(dialog, text=message, font=message_font, wraplength=350, justify="left")
        message_label.pack(pady=20, padx=20)

        # Create Yes/No buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)

        result = {"value": False}

        def on_yes():
            result["value"] = True
            dialog.destroy()

        def on_no():
            result["value"] = False
            dialog.destroy()

        yes_button = tk.Button(button_frame, text="Yes", command=on_yes, font=button_font, bg="#4186DD", fg="#ffffff", width=10, activebackground="#366EB8")
        no_button = tk.Button(button_frame, text="No", command=on_no, font=button_font, bg="#E57373", fg="#ffffff", width=10, activebackground="#D64D4D")

        yes_button.grid(row=0, column=0, padx=10)
        no_button.grid(row=0, column=1, padx=10)

        self.root.wait_window(dialog)
        return result["value"]

    def update_log_display(self, text):
        """Clears the display and inserts new text with colored formatting."""
        self.log_display.config(state=tk.NORMAL)
        self.log_display.delete(1.0, tk.END)  # Clear old content

        # Apply colored tags based on content
        for line in text.splitlines():
            # Split the line into segments, and apply colors only to specific elements
            start_index = 0
            for match in re.finditer(r'(\d+)|(\'.*?\')', line):
                matched_text = match.group()
                match_start, match_end = match.start(), match.end()

                # Insert text before the match
                if start_index < match_start:
                    self.log_display.insert(tk.END, line[start_index:match_start])

                # Insert the matched text with specific color tags
                if re.match(r'\d+', matched_text):
                    self.log_display.insert(tk.END, matched_text, "number")
                elif re.match(r'\'.*?\'', matched_text):
                    self.log_display.insert(tk.END, matched_text, "quoted")

                # Update start index for the next iteration
                start_index = match_end

            # Insert remaining text after the last match
            if start_index < len(line):
                self.log_display.insert(tk.END, line[start_index:])

            # Add a newline after each line
            self.log_display.insert(tk.END, "\n")

        # Define tag colors
        self.log_display.tag_configure("number", foreground="#005BBB")  # Darker blue for numbers
        self.log_display.tag_configure("quoted", foreground="#E67E22")  # Deeper orange for quoted text

        self.log_display.config(state=tk.DISABLED)

    def clean_pid_docs_and_empty_logs(self):
        """
        Clean up the pid_docs directory by removing files and the directory itself.

        Clean up the logs directory if no log file exits.
        """
        # Remove main_pid.txt if it exists
        if os.path.exists("../pid_docs/main_pid.txt"):
            os.remove("../pid_docs/main_pid.txt")
        # Remove script_name.txt if it exists
        if os.path.exists("../pid_docs/script_name.txt"):
            os.remove("../pid_docs/script_name.txt")
        # Remove pid_docs directory if it exists and is empty
        if os.path.exists("../pid_docs") and os.path.isdir("../pid_docs"):
            try:
                os.rmdir("../pid_docs")
            except OSError:
                pass  # Directory not empty or other error
        self.clean_directory("../logs")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProgramManager(root)
    app.check_status_and_show_log()
    root.mainloop()

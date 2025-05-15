import os
import time
import torch
import cv2
import logging
import sys
import shutil
import csv
import numpy as np
from ultralytics import YOLO

def process_videos(input_dir, video_output_dir, model_path, tracker_config,bbox_output_dir,show_video=False):
    # Load the YOLO model with tracking enabled
    if torch.cuda.is_available():
        print("Using CUDA")
        model = YOLO(model_path).to('cuda')  # Use GPU for faster processing
    else:
        print("CUDA unavailable, Using CPU")
        model = YOLO(model_path)
    print("")
    
    video_amount = get_video_amount(input_dir)
    processed_video = 0
    print(f"Found {video_amount} videos in total.")
    print("")

    # clear or create output directory if it doesn't exist
    clear_directory(video_output_dir)
    clear_directory(bbox_output_dir)

    # Predefined color for drawing bounding boxes
    bbox_color = (0, 255, 0)  # Fluorescent Green

    # Loop through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for video_file in files:
            if not is_video_file(video_file):
                continue

            start_time = time.time()
            print(f"({processed_video + 1}/{video_amount}) Processing '{video_file}'...")

            video_path = os.path.join(root, video_file)

            # Compute the relative path from the input directory to the current file's directory
            relative_path = os.path.relpath(root, input_dir)

            # Create the corresponding subdirectory in the output directory
            output_subdir = os.path.join(video_output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
        
            # Define the output file path
            output_path = os.path.join(output_subdir, video_file)
            
            # Calculate the motion level of the video
            try:
                motion_level = compute_motion_level(video_path)
            except Exception as e:
                print(f"Error computing motion for {video_file}: {e}")
                # Use default value if motion calculation fails
                MIN_BOX_AREA_RATIO = 0.15

            # Get video properties
            frame_width, frame_height, frame_fps = get_video_properties(video_path)
            frame_area = frame_width * frame_height
            
            #Set suitabe parameters to the processed video
            if frame_fps > 0 and frame_fps <= 25:
                fps = 'low'
            elif frame_fps > 25 and frame_fps <= 50:
                fps = 'mid'
            elif frame_fps > 50 and frame_fps <= 100:
                fps = 'high'
            elif frame_fps > 100:
                fps = 'extra high'
            else:
                fps = 'default'
                
            match motion_level:
                case 'low':
                    MIN_DETECTIONS = 3
                    MIN_BOX_AREA_RATIO = 0.1
                case 'high':
                    MIN_DETECTIONS = 2
                    MIN_BOX_AREA_RATIO = 0.2
                case _:
                    MIN_DETECTIONS = 2
                    MIN_BOX_AREA_RATIO = 0.15
                    print(f"WARNING: use default MIN_DETECTIONS and \
MIN_BOX_AREA_RATIO to process '{video_file}'")
                
            match motion_level, fps:
                case 'low','low':
                    FRAME_INTERVAL = 4
                case 'low','mid':
                    FRAME_INTERVAL = 5
                case 'low','high':
                    FRAME_INTERVAL = 10
                case 'low', 'extra high':
                    FRAME_INTERVAL = 20
                case 'high','low':
                    FRAME_INTERVAL = 2
                case 'high','mid':
                    FRAME_INTERVAL = 3
                case 'high','high':
                    FRAME_INTERVAL = 6
                case 'high', 'extra high':
                    FRAME_INTERVAL = 10
                case _:
                    FRAME_INTERVAL = 3
                    print(f"WARNING: use default parameters to process '{video_file}'")
                    
            print(f"Input_FPS:  {round(frame_fps)} \n\
Motion Classification:  {motion_level} \n\
Adaptive Parameters: \n\
    FRAME_INTERVAL:     {FRAME_INTERVAL} \n\
    MIN_DETECTIONS:     {MIN_DETECTIONS} \n\
    MIN_BOX_AREA_RATIO: {MIN_BOX_AREA_RATIO} \n\
Tracking person...")

            # Initialize video writer
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'avc1'),
                frame_fps,
                (frame_width, frame_height)
            )

            # Initialize tracking variables
            frame_counter = 0
            selected_person_id = None
            draw_only_selected = False
            previous_boxes = None  # To store previous detection boxes
            current_selected_person = None  # To store the selected person
            id_mapping = {}
            next_available_id = 1
            detection_counts = {}  # To store detection counts for each person

            # Use model.track() to process the entire video with stream=True to maintain tracker state
            try:
                results_generator = model.track(
                    source=video_path,
                    stream=True,
                    tracker=tracker_config,
                    verbose=False
                )
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

            # Loop through the generated results
            for results in results_generator:
                frame = results.orig_img  # Get the original frame
                frame_counter += 1  # Increment the frame counter

                # Only process detection every FRAME_INTERVAL frames
                if frame_counter % FRAME_INTERVAL != 0:
                    # If previous boxes exist, draw them on the current frame
                    if previous_boxes is not None:
                        if draw_only_selected and selected_person_id is not None and current_selected_person is not None:
                            # Find the box corresponding to the selected person
                            for box in previous_boxes:
                                if box.id is not None:
                                    original_id = int(box.id[0])
                                    mapped_id = id_mapping.get(original_id)
                                    if mapped_id == selected_person_id:
                                        draw_person(frame, box, bbox_color, mapped_id, detection_counts, MIN_DETECTIONS,video_file,bbox_output_dir,frame_counter)
                                        break
                        else:
                            # Draw all previous boxes
                            draw_all_persons(frame, previous_boxes, bbox_color, id_mapping, detection_counts, MIN_DETECTIONS,video_file,bbox_output_dir,frame_counter)
                    # Write the frame to output
                    out.write(frame)
                    if show_video:
                        if display_frame(frame):
                            break
                    continue

                # Proceed to process the results
                if results and results.boxes:
                    # Filter out bounding boxes smaller than MIN_BOX_AREA_RATIO of the frame area
                    filtered_boxes = []
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_area = (x2 - x1) * (y2 - y1)
                        if bbox_area < MIN_BOX_AREA_RATIO * frame_area:
                            continue  # Skip boxes with too small area
                        filtered_boxes.append(box)

                    # Update previous_boxes with the filtered boxes
                    previous_boxes = filtered_boxes if filtered_boxes else None

                    # Reset current_selected_person
                    current_selected_person = None

                    # Update ID mapping and detection counts for filtered_boxes
                    for box in filtered_boxes:
                        if box.id is not None:
                            original_id = int(box.id[0])
                            if original_id not in id_mapping:
                                id_mapping[original_id] = next_available_id
                                detection_counts[next_available_id] = 0  # Initialize detection count
                                next_available_id += 1
                            mapped_id = id_mapping[original_id]
                            # Increment detection count
                            detection_counts[mapped_id] += 1

                    if draw_only_selected and selected_person_id is not None:
                        # Check if selected person is still detected
                        person_found = False
                        for track in filtered_boxes:
                            if int(track.cls[0]) == 0 and float(track.conf[0]) > 0.55 and track.id is not None:
                                original_id = int(track.id[0])
                                mapped_id = id_mapping.get(original_id)
                                if mapped_id == selected_person_id:
                                    person_found = True
                                    current_selected_person = track
                                    break
                        if not person_found:
                            # Person not found, reset
                            draw_only_selected = False
                            selected_person_id = None
                            current_selected_person = None
                        else:
                            # Check if the person still occupies more than 50% of the frame
                            x1, y1, x2, y2 = map(int, current_selected_person.xyxy[0])
                            bbox_area = (x2 - x1) * (y2 - y1)
                            if bbox_area < 0.5 * frame_area:
                                # Reset if person no longer occupies enough area
                                draw_only_selected = False
                                selected_person_id = None
                                current_selected_person = None
                    else:
                        # Find the person with the largest bounding box
                        max_bbox_area = 0
                        largest_person = None
                        for track in filtered_boxes:
                            if (
                                int(track.cls[0]) == 0 and
                                float(track.conf[0]) > 0.55 and
                                track.id is not None  # Ensure ID is assigned
                            ):
                                x1, y1, x2, y2 = map(int, track.xyxy[0])
                                bbox_area = (x2 - x1) * (y2 - y1)
                                if bbox_area > max_bbox_area:
                                    max_bbox_area = bbox_area
                                    largest_person = track

                        if largest_person is not None:
                            if max_bbox_area > 0.5 * frame_area:
                                original_id = int(largest_person.id[0])
                                mapped_id = id_mapping.get(original_id)
                                selected_person_id = mapped_id
                                draw_only_selected = True
                                current_selected_person = largest_person
                            else:
                                # Largest person doesn't occupy enough area
                                selected_person_id = None
                                draw_only_selected = False
                        else:
                            # No person with assigned ID detected
                            selected_person_id = None
                            draw_only_selected = False
                            previous_boxes = None  # Reset previous boxes if no detections

                    # Now, decide what to draw
                    if draw_only_selected and selected_person_id is not None and current_selected_person is not None:
                        # Draw only the selected person
                        draw_person(frame, current_selected_person, bbox_color, selected_person_id, detection_counts, MIN_DETECTIONS,video_file,bbox_output_dir,frame_counter)
                    else:
                        # Draw bounding boxes for all persons
                        draw_all_persons(frame, filtered_boxes, bbox_color, id_mapping, detection_counts, MIN_DETECTIONS,video_file,bbox_output_dir,frame_counter)
                else:
                    # No detections
                    if draw_only_selected and selected_person_id is not None:
                        # Since person is not detected, reset
                        draw_only_selected = False
                        selected_person_id = None
                        current_selected_person = None
                    previous_boxes = None  # Reset previous boxes

                # Write the frame to the output video
                out.write(frame)

                # Display the annotated frame (optional)
                if show_video:
                    if display_frame(frame):
                        break

            # Release resources and print processing time
            out.release()
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                time_display = f"{elapsed_time:.2f} seconds"
            else:
                minutes, seconds = divmod(elapsed_time, 60)
                time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"

            print(f"Completed '{video_file}' in {time_display}, total frames: {frame_counter}")
            print("")
            processed_video += 1  # Increment counter after processing each video

    if show_video:
        cv2.destroyAllWindows()

def is_video_file(filename):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return filename.lower().endswith(video_extensions)

def get_video_amount(input_dir):
    """
    Count the total number of video files (.mp4) in the input directory and its subdirectories.

    Args:
        input_dir (str): Path to the input directory.

    Returns:
        int: Total number of video files.
    """
    total_count = 0
    for root, _, files in os.walk(input_dir):
        for video_file in files:
            if is_video_file(video_file):
                total_count += 1
    return total_count

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_width, frame_height, frame_fps

def draw_person(frame, person, color, mapped_id, detection_counts, min_detections,video_file,bbox_output_dir,frame_counter):
    if person is None:
        return  # Avoid drawing if person is None

    # Check if the person has been detected enough times
    if detection_counts.get(mapped_id, 0) < min_detections:
        return  # Do not draw if detected fewer times than min_detections

    x1, y1, x2, y2 = map(int, person.xyxy[0])
    confidence = float(person.conf[0])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Determine label content based on box width
    full_label = f"Person ID: {mapped_id}, Conf: {confidence:.2f}" if mapped_id is not None else "Person ID: N/A, " + f"Conf: {confidence:.2f}"
    id_only_label = f"Person ID: {mapped_id}" if mapped_id is not None else "Person ID: N/A"

    # Calculate text sizes
    full_text_size, _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_width, text_height = full_text_size

    # Check if the full label fits within the rectangle's width
    label = full_label if (x2 - x1) > text_width + 10 else id_only_label  # 10 pixels as margin

    # Set label position with a 5-pixel offset
    label_x = x1 + 5
    label_y = y1 - 10 if y1 > 23 else y1 + text_height + 5

    # Ensure text does not go beyond the rectangle when positioned inside
    if label_y + text_height > y2:
        label_y = y2 - 5

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )
    
    #Write down the bounding box info
    record_bounding_box(bbox_output_dir,video_file,frame_counter,mapped_id,x1,y1,x2,y2)

def draw_all_persons(frame, boxes, color, id_mapping, detection_counts, min_detections,video_file,bbox_output_dir,frame_counter):
    for box in boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.55:
            if box.id is None:
                continue  # Skip if ID is not assigned
            original_id = int(box.id[0])
            mapped_id = id_mapping.get(original_id)
            if mapped_id is None:
                continue  # Skip if ID is not in mapping

            # Check if the person has been detected enough times
            if detection_counts.get(mapped_id, 0) < min_detections:
                continue  # Do not draw if detected fewer times than min_detections

            # Use the draw_person function to draw each person
            draw_person(frame, box, color, mapped_id, detection_counts, min_detections,video_file,bbox_output_dir,frame_counter)

def display_frame(frame, title="YOLO Person Detection with Tracking"):
    cv2.imshow(title, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    return False

def compute_motion_level(video_path, sample_frames=30, threshold=95):
    """
    Calculate the motion level of a video using sparse optical flow.

    Parameters:
        video_path (str): Path to the video file.
        sample_frames (int): Number of frames per group for motion calculation.
        threshold (int): Motion threshold to classify as 'high' or 'low'.

    Returns:
        str: 'high' for high motion, 'low' for low motion.
    """
    print("Calcutating motion level...")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"No frames found in '{video_path}'. Returning 'high'.")
            cap.release()
            return 'high'

        # Determine sampling strategy
        if total_frames < sample_frames:
            num_groups = 1
            group_size = total_frames
        else:
            # Aim to sample approximately 10% of the total frames
            total_sample_frames = max(int(0.1 * total_frames), sample_frames)
            num_groups = max(total_sample_frames // sample_frames, 1)
            group_size = sample_frames

        # Calculate the interval between the start of each group to distribute them evenly
        interval = max(total_frames // num_groups, group_size)

        motion_sum = 0
        frame_count = 0

        for g in range(num_groups):
            # Set the position to the start of the current group
            start_frame = g * interval
            if start_frame >= total_frames:
                break  # Prevent seeking beyond the total frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read the first frame of the group
            ret, prev_frame = cap.read()
            if not ret:
                print(f"Failed to read frame at position {start_frame}. Skipping group {g+1}.")
                continue
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Detect feature points in the first frame using Shi-Tomasi corner detection
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is None:
                print(f"No features found in group {g+1}. Skipping this group.")
                continue  # Skip if no features are found

            for i in range(1, group_size):
                ret, frame = cap.read()
                if not ret:
                    print(f"Reached end of video while processing group {g+1}.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate optical flow using Lucas-Kanade method
                lk_params = dict(winSize=(15, 15),
                                 maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

                if p1 is None or st is None:
                    print(f"Optical flow failed for group {g+1}, frame {i}.")
                    break

                # Select only the good points
                good_new = p1[st.flatten() == 1]
                good_old = p0[st.flatten() == 1]

                if len(good_new) == 0:
                    print(f"No good points tracked in group {g+1}, frame {i}.")
                    break

                # Compute the motion as the sum of Euclidean distances of all good points
                motion = np.sum(np.linalg.norm(good_new - good_old, axis=1))
                motion_sum += motion
                frame_count += 1

                # Prepare for the next iteration
                prev_gray = gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

        cap.release()

        if frame_count == 0:
            print(f"No motion detected in '{video_path}'. Returning 'low'.")
            return 'low'

        # Calculate average motion
        average_motion = motion_sum / frame_count
        print(f"Average motion: {round(average_motion)}")

        return 'high' if average_motion > threshold else 'low'

    except Exception as e:
        print(f"An error occurred: {e}")
        return 'high'

def setup_logging(log_filename="../logs/trackPerson.log"):
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

def record_bounding_box(output_dir, video_id, frame_id, person_id, xmin, ymin, xmax, ymax, file_name="bounding_box.csv"):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(output_dir, file_name)

    # Check if the file already exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow(['video_id', 'frame_id', 'person_id',
                            'xmin', 'ymin', 'xmax', 'ymax'])

        # Write the data row
        writer.writerow([video_id, frame_id, person_id, xmin, ymin, xmax, ymax])

def clear_directory(dir):
    """
    Clear the output directory by removing its contents and recreating it.
    
    Args:
        dir (str): Path to the output directory.
    """
    if os.path.exists(dir):
        print(f"{dir} not empty, clearing...")
        print("")
        shutil.rmtree(dir)
    os.makedirs(dir)

def run(logfile = False,SHOW_VIDEO = False):
    """
    Run the process to annotate persons in videos.

    Args:
        logfile (bool): Whether to generate a log file. Defaults to True.
        SHOW_VIDEO (bool): Whether to display the video with annotations in real-time. Defaults to False.

    Returns:
        None
    """
    # Configuration parameters
    INPUT_DIR = "../processed_videos/"
    VIDEO_OUTPUT_DIR = "../annonated_videos/"
    MODEL_PATH = "../yolo model/yolo11n.pt"
    TRACKER_CONFIG = "../yolo model/botsort.yaml"
    BOUNDING_BOX_OUTPUT_DIR = "../bounding_box/"

    if logfile:
        setup_logging()

    print("Start Program: Track Person ...")
    start_time = time.time()
    process_videos(INPUT_DIR, VIDEO_OUTPUT_DIR, MODEL_PATH, TRACKER_CONFIG,BOUNDING_BOX_OUTPUT_DIR,SHOW_VIDEO)
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time < 60:
        time_display = f"{elapsed_time:.2f} seconds"
    else:
        minutes, seconds = divmod(elapsed_time, 60)
        time_display = f"{int(minutes)} minutes {seconds:.2f} seconds"
    print(f"track person finished. Total time: {time_display}.")
    print("")

if __name__ == "__main__":
    run()
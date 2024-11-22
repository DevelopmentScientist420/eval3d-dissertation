import cv2
import os


def extract_frames(video_path, output_dir, n_div=12):
    """
    Extract frames from a video at the specified times.

    Parameters:
    - video_path: Path to the video file.
    - times: List of times in seconds at which to extract frames.

    Returns:
    - A list of numpy arrays, each representing an extracted frame.
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return []
        
    os.makedirs(output_dir, exist_ok=True)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the number of frames in the video
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    frames = []
    
    frame_indexes = [i for i in range(0, len_frames, len_frames//n_div)]
    
    for angle_idx, frame_index in enumerate(frame_indexes):
        # Set the video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        
        # If the frame was read successfully, add it to the list
        if ret:
            save_path = os.path.join(output_dir, f"{angle_idx}.png")
            cv2.imwrite(save_path, frame)
            frames.append(save_path)
        else:
            print(f"Error: Could not read frame at time {time} seconds for video {video_path}")
    
    # Release the video capture object
    cap.release()
    
    return frames
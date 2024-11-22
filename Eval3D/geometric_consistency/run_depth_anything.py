import os
import glob
import subprocess
import argparse

def read_data(args):
    data = {}
    prompt_idx = 0

    # Traverse object directories
    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag = 0
        old_flag = 0
        
        # Gather data for RGB and depth images
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/rgb_images/*.png')))
        if len(data_folder_content) > 0:
            data_flag = 1

        rgb_data_folder_content = []
        for img_path in data_folder_content:
            if "rgba" in img_path: continue
            rgb_data_folder_content.append(img_path)
        data_folder_content = rgb_data_folder_content
        
        depth_anything_folder = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/depth_anything/*.png')))
        
        # Skip if 120 depth images are already present
        if len(depth_anything_folder) == 120:
            prompt_idx += 1
            continue
        
        if data_flag:
            image_key = data_folder_content[0].split('@')[0]
            
            if image_key in data:
                curr_prompt_idx = prompt_idx
                old_flag = 1
                prompt_idx = data[image_key][0]
            
            data[image_key] = (prompt_idx, os.path.join("/", "/".join(data_folder_content[0].split('/')[1:-1])))

            print(f"Found {len(data_folder_content)} images. Prompt ID: {prompt_idx}, Path: {data[image_key][1]}")
            
            if old_flag:
                prompt_idx = curr_prompt_idx
            else:
                prompt_idx += 1

    return data


def is_task_done(data_path):
    """Check if a task (job) is completed based on the existence of an output file."""
    return os.path.exists(data_path.replace('rgb_images', 'depth_anything') + '/0119_depth_anything_rendered.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the data')
    parser.add_argument('--algorithm_name', type=str, required=True, help='Algorithm name folder')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for prompt processing')
    parser.add_argument('--end_idx', type=int, default=10, help='End index for prompt processing')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs available')
    parser.add_argument('--available_gpus', type=str, required=True, help='Comma-separated list of available GPU IDs (e.g., "1,2,3")')
    parser.add_argument('--tmux_id', type=int, default=1, help='start tmux id')
    
    args = parser.parse_args()

    # Parse available GPUs into a list of integers
    available_gpus = [int(gpu) for gpu in args.available_gpus.split(',')]

    # Read data
    algorithm_data = read_data(args)

    # Initialize variables for managing parallel tasks
    queue_images = []  # Store jobs in queue
    free_gpus = args.num_gpus  # Number of available GPUs
    curr_gpu_id = 0  # Keep track of which GPU to assign next
    tmux_id = args.tmux_id  # Tmux session ID counter

    idx = args.start_idx
    while idx < len(algorithm_data) or len(queue_images)>0:
        # Check if any task is completed and free up GPU
        removal_idx = []
        for img_idx, (image_path, tmux_idx) in enumerate(queue_images):
            if is_task_done(image_path):
                print(f'{image_path.replace("rgb_images", "depth_anything")}/0119_depth_anything_rendered.txt exists, task complete')
                removal_idx.append((img_idx, tmux_idx))

        # Remove completed tasks from queue
        for img_idx, tmux_idx in removal_idx:
            _ = queue_images.pop(img_idx)
            subprocess.run(["tmux", "kill-session", "-t", str(tmux_idx)])
            free_gpus += 1  # Increment available GPU count

        # If we have free GPUs, assign the next task
        if free_gpus > 0 and idx < len(algorithm_data):
            try:
                datum_key = list(algorithm_data.keys())[idx]
                datum = algorithm_data[datum_key]
                prompt_idx = int(datum[0])

                # Ensure the prompt index is within the provided range
                if not (args.start_idx <= prompt_idx < args.end_idx):
                    idx += 1
                    continue

                data_path = datum[1].replace("'", "\\'")
                gpu_id = available_gpus[curr_gpu_id % args.num_gpus]  # Assign GPU

                print(f'DEPTH-ANYTHING | GPU: {gpu_id} | PROMPT ID: {prompt_idx} | DATA_PATH: {data_path} | TMUX_ID: {tmux_id}')

                # Construct the command for processing this task
                command = f'''cd Depth-Anything/; CUDA_VISIBLE_DEVICES={gpu_id} python run.py --encoder vitl --img-path {data_path} --outdir {data_path.replace('rgb_images', 'depth_anything')}'''
                command += "; sleep infinity"  # Keep the process alive

                # Launch the command in a new tmux session
                subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id), command])

                # Update tracking variables
                curr_gpu_id += 1
                queue_images.append((data_path, tmux_id))
                free_gpus -= 1  # Decrease the count of available GPUs
                tmux_id += 1
                idx += 1  # Move to the next task in the queue

            except Exception as e:
                print('Exception:', e)
                pass
        
    print("All tasks are launched.")

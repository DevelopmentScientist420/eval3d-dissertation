import os
import glob
import argparse
import tqdm
import cv2
import numpy as np
from PIL import Image

def get_algorithm_data(args):
    data = {}
    prompt_idx = 0

    for idx, object_id in enumerate(os.listdir(os.path.join(args.base_dir, args.algorithm_name))):
        data_flag = 0
        old_flag = 0
        
        data_folder_content = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/rgb_images/*.png')))
        if len(data_folder_content) > 0:
            data_flag = 1

        depth_anything_folder = sorted(glob.glob(os.path.join(args.base_dir, args.algorithm_name, object_id, 'save/*/depth_anything_normal_camera/*.png')))
        
        # # Skip if folder already exists
        # if len(depth_anything_folder) == 60:
        #     prompt_idx += 1
        #     continue

        # Process valid data
        if data_flag:
            image_key = data_folder_content[0].split('@')[0]
            
            if image_key in data:
                curr_prompt_idx = prompt_idx
                old_flag = 1
                prompt_idx = data[image_key][0]

            data[image_key] = (prompt_idx, os.path.join("/", "/".join(data_folder_content[0].split('/')[1:-2])))
            print(f"Found {len(data_folder_content)} images. Prompt ID: {prompt_idx}, Path: {data[image_key][1]}")
            
            if old_flag:
                prompt_idx = curr_prompt_idx
            else:
                prompt_idx += 1

    return data


def get_prompt_data(data_dir):
    all_normal_data = sorted(glob.glob(os.path.join(data_dir, "normal_world", "*.npy")))
    all_batch_data = sorted(glob.glob(os.path.join(data_dir, "batch_data", "*.npy")))
    all_opacity_data = sorted(glob.glob(os.path.join(data_dir, "opacity", "*.png")))
    all_rgb_data = sorted(glob.glob(os.path.join(data_dir, "rgb_images", "*.png")))
    all_depth_anything = sorted(glob.glob(os.path.join(data_dir, "depth_anything", "*.npy")))
    
    selected_rgb_data = []
    for img in all_rgb_data:
        if 'rgba' in img: continue
        selected_rgb_data.append(img)
    all_rgb_data = selected_rgb_data
    
    return all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything


def normalize_vectors(vectors, axis=-1, eps=1e-8):
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / (norms + eps)

def get_normal_transformed(normal, transformation):
    normal = normal.reshape(-1, 3)
    normal_transformed = transformation @ normal.transpose()
    normal_transformed = normal_transformed.swapaxes(0, 1)
    normal_transformed = normal_transformed.reshape(512, 512, 3)
    normal_transformed = normal_transformed[:,:,:3]
    return normal_transformed

def depth_map_to_normal_map(depth_map):
    depth_map *= -1

    # Calculate gradients using Sobel operator
    gradient_y = np.gradient(depth_map, axis=0) 
    gradient_x = np.gradient(depth_map, axis=1) 

    # Calculate surface normals
    normal_x = gradient_x
    normal_y = gradient_y
    normal_z = np.ones_like(depth_map)

    # Normalize normals
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
    return normal_map



def compute_metric(data_path, normal_metric_threshold=0.3, opacity_threshold=0.01):
    os.system('''mkdir -p ''' + os.path.join(data_path.replace("'", "\\'"), "normal_metric"))
    os.system('''mkdir -p ''' + os.path.join(data_path.replace("'", "\\'"), "normal_camera"))
    os.system('''mkdir -p ''' + os.path.join(data_path.replace("'", "\\'"), "depth_anything_normal_camera"))

    all_normal_data, all_batch_data, all_opacity_data, all_rgb_data, all_depth_anything = \
        get_prompt_data(data_path)

    assert(len(all_normal_data)==len(all_batch_data))
    assert(len(all_normal_data)==len(all_rgb_data))
    
    all_opacity = []
    geometric_consistency_metric = []
    for idx in tqdm.tqdm(range(len(all_normal_data))):
        # we compute metric using every second of the rendered 120 frames for compute reason
        if idx %2!=0: continue
        
        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        normal_world = np.load(all_normal_data[idx])
        rgb_image = np.asarray(Image.open(all_rgb_data[idx]))
        depth_anything = np.load(all_depth_anything[idx])
        opacity_map = np.asarray(Image.open(all_opacity_data[idx]))[...,0] / 255.
        normal_world = cv2.resize(normal_world, (512, 512))
        depth_anything = cv2.resize(depth_anything, (512, 512))
        rgb_image = cv2.resize(rgb_image, (512, 512))
        opacity_map = cv2.resize(opacity_map, (512, 512))
        opacity_map = (opacity_map>0)*1.

        depth_anything_normal_camera = depth_map_to_normal_map(depth_anything)
            
        c2w = np.array(batch_data['c2w'].cpu().numpy()[:,:3,:3])
        w2c = np.linalg.inv(c2w[0])

        # threestudio normal adjustment
        normal_world = (normal_world * 2.) - 1
        normal_world = normalize_vectors(normal_world)
        normal_camera = get_normal_transformed(normal_world, w2c)
        normal_camera = normalize_vectors(normal_camera)
        normal_camera = (normal_camera + 1.0)/2.
        normal_camera = normal_camera * opacity_map[...,None]
        
        # from: https://github.com/deepseek-ai/DreamCraft3D/blob/b20d9386198b3965c78ba71c98156628fc41ecd3/threestudio/systems/dreamcraft3d.py#L176
        normal_camera[..., 0] = 1 - normal_camera[..., 0]
        normal_camera = (2 * normal_camera - 1)
        normal_camera = (normal_camera + 1.) /2.
        normal_camera = normalize_vectors(normal_camera)
        normal_camera = normal_camera * opacity_map[...,None]
        
        # depth anything normal adjustment
        depth_anything_normal_camera[...,-1] *= -1
        depth_anything_normal_camera = (depth_anything_normal_camera + 1.) / 2.
        depth_anything_normal_camera = (1 - 2 * depth_anything_normal_camera)  # [B, 3]
        depth_anything_normal_camera = (depth_anything_normal_camera + 1.) /2.
        depth_anything_normal_camera = normalize_vectors(depth_anything_normal_camera)
        depth_anything_normal_camera = depth_anything_normal_camera * opacity_map[...,None]

        normal_metric = np.arccos(np.clip(np.sum(depth_anything_normal_camera.reshape(-1,3) * normal_camera.reshape(-1,3), axis=-1).reshape(512, 512), -1., 1.))
        normal_metric = normal_metric * opacity_map
        
        np.save(all_batch_data[idx].replace('batch_data', 'normal_camera'), np.asarray(normal_camera))
        np.save(all_batch_data[idx].replace('batch_data', 'depth_anything_normal_camera'), np.asarray(depth_anything_normal_camera))
        Image.fromarray(np.asarray(normal_camera*255, dtype=np.uint8)).save(all_rgb_data[idx].replace('rgb_images', 'normal_camera'))
        Image.fromarray(np.asarray(depth_anything_normal_camera*255, dtype=np.uint8)).save(all_rgb_data[idx].replace('rgb_images', 'depth_anything_normal_camera'))
        np.save(all_batch_data[idx].replace('batch_data', 'normal_metric'), normal_metric)

        all_opacity.append(opacity_map.sum()/np.prod(opacity_map.shape))


        from scipy.ndimage import uniform_filter
        kernel_size = 11
        normal_metric = uniform_filter(normal_metric, size=kernel_size)
        normal_metric = normal_metric * opacity_map
        mask = ((normal_metric<normal_metric_threshold) | np.isnan(normal_metric))
        geometric_consistency_metric.append((1-mask).sum() / (opacity_map.sum()+1.e-8))
        

    # Reconstructions with very small (smaller than opacity_threshold) rendered mask / opacity are considered 100% faulty.
    mean_opacity = np.stack(all_opacity).mean()
    geometric_consistency_metric = np.stack(geometric_consistency_metric).reshape(-1, 3).max(axis=-1).mean(axis=0)
    geometric_consistency_metric = 100. if mean_opacity < opacity_threshold else 100 * geometric_consistency_metric
    geometric_consistency_metric = 100 - geometric_consistency_metric
    with open(os.path.join(data_path, "geometric_consistency_metric.txt"), "w") as f:
        f.write("Geometric Consistency Metric: {}".format(geometric_consistency_metric))
    f.close()
    
    return geometric_consistency_metric
    


def main(args):
    algorithm_data = get_algorithm_data(args)
    for data_key, datum in algorithm_data.items():
        print("============== Starting Metric Computation ====================")
        prompt_idx = int(datum[0])
        data_path = datum[1]
        print('''GEOMETRIC CONSISTENCY METRIC | PROMPT ID: {} | DATA_PATH: {}'''.format(prompt_idx, data_path))
        geometric_consistency_metric = compute_metric(data_path)
        print('''GEOMETRIC CONSISTENCY METRIC | PROMPT ID: {} | METRIC: {}'''.format(prompt_idx, geometric_consistency_metric))


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for Geometric Consistency Computation')
    parser.add_argument('--base_dir', required=True, type=str, help='Base directory path')
    parser.add_argument('--algorithm_name', required=True, type=str, help='Name of the algorithm')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
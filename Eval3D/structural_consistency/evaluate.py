import argparse
import os
import subprocess
import glob
import uuid
import cv2
import tqdm
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dreamsim import dreamsim
from preprocess import preprocess_data

def parse_args():
    parser = argparse.ArgumentParser(description="Parse input paths.")
    parser.add_argument('--prompt_id', type=str, default=None, help='Path to the prompt data.')
    parser.add_argument('--base_data_path', type=str, default=None, help='Path to the algorithm data.')
    parser.add_argument('--algorithm_name', type=str, default=None, help='')
    parser.add_argument('--tmux_id', type=int, default=1, help='')
    args = parser.parse_args()
    args.algorithm_data_path = os.path.join(args.base_data_path, args.algorithm_name)
    return args

def get_prompt_data(prompt_data_path):
    all_opacity_data = sorted(glob.glob(os.path.join(prompt_data_path, "opacity", "*.png")))
    assert len(all_opacity_data)==120, f"Length of all_opacity_data={len(all_opacity_data)} !=120"
    return all_opacity_data

def compute_dreamsim(img_0_path, img_ref_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = dreamsim(pretrained=True, device=device)

    algo_img = np.asarray(Image.open(img_ref_path))
    mask = (algo_img[...,3:]>0)*1
    algo_img = algo_img[...,:3] * mask + (1-mask)*255
    algo_img = Image.fromarray(np.asarray(algo_img, dtype=np.uint8))
    
    img_ref = preprocess(algo_img).to(device)
    img_0 = preprocess(Image.open(img_0_path)).to(device)

    d0 = model(img_ref, img_0)
    return d0.item()



def compute_metric(reconstructed_asset_path, zero123_generation_path, opacity_threshold=0.01):
    """
        Requires three set of multi-view data --
        - 3D metadata from corresponding text-3D algorithm: Rendered RGB images at 0, 90, 180, 270 degree viewpoints.
        - Zero123 NVS outputs corresponding to 0 degree rendered image
        - Zero123 NVS outputs corresponding to 90 degree rendered image
    """

    all_opacity_data = get_prompt_data(glob.glob(os.path.join(reconstructed_asset_path, "save/it*-test"))[0])
    zero123_data_path_0 = os.path.join(zero123_generation_path, "0000_cropped_rgba.png*/save/it0-test/rgb_images/")
    zero123_data_path_90 = os.path.join(zero123_generation_path, "0030_cropped_rgba.png*/save/it0-test/rgb_images/")
    zero123_data_path_0 = sorted(glob.glob(zero123_data_path_0))[-1]
    zero123_data_path_90 = sorted(glob.glob(zero123_data_path_90))[-1]

    zero123_data_path_0 = os.path.join(zero123_data_path_0, "*.png")        
    zero123_data_path_90 = os.path.join(zero123_data_path_90, "*.png")
    all_zero123_images_0 = sorted(glob.glob(zero123_data_path_0))
    all_zero123_images_90 = sorted(glob.glob(zero123_data_path_90))
    
    # aligns all_zero123_images_90 images with all_zero123_images_0
    all_zero123_images_90 = all_zero123_images_90[-2:-1] + all_zero123_images_90[0:-2] + all_zero123_images_90[-1:]
    
    reconstructed_asset_path = os.path.join(reconstructed_asset_path, "save/it*-test/rgb_images/")
    reconstructed_asset_path = sorted(glob.glob(reconstructed_asset_path))[-1]
    reconstructed_asset_path = os.path.join(reconstructed_asset_path, "*_cropped_rgba.png")
    all_reconstructed_asset_images = sorted(glob.glob(reconstructed_asset_path))
    
    
    vis_image_paths_zero123 = [all_zero123_images_0[0], all_zero123_images_0[1], all_zero123_images_0[2], all_zero123_images_0[3]]
    vis_image_paths_zero123_90 = [all_zero123_images_90[0], all_zero123_images_90[1], all_zero123_images_90[2], all_zero123_images_90[3]]
    vis_image_paths_algorithm = [all_reconstructed_asset_images[0], all_reconstructed_asset_images[30], all_reconstructed_asset_images[60], all_reconstructed_asset_images[90]]

    img_h = 256
    dreamsim_scores = []
    zero123_img_canvas = np.zeros((img_h*3, img_h*len(vis_image_paths_zero123), 3), dtype=np.uint8)
    for img_idx, (zero123_img_file, zero123_90shift_img_file, algo_img_file) in enumerate(zip(vis_image_paths_zero123, vis_image_paths_zero123_90, vis_image_paths_algorithm)):
        
        algo_img = np.asarray(Image.open(algo_img_file))
        mask = (algo_img[...,3:]>0)*1
        algo_img = algo_img[...,:3] * mask + (1-mask)*255
        algo_img = Image.fromarray(np.asarray(algo_img, dtype=np.uint8)).resize((256,256))

        zero123_img = np.asarray(Image.open(zero123_img_file))
        zero123_img_90shift = np.asarray(Image.open(zero123_90shift_img_file))

        zero123_img_canvas[0*img_h:1*img_h, img_h*img_idx:img_h*(img_idx+1)] = zero123_img
        zero123_img_canvas[1*img_h:2*img_h, img_h*img_idx:img_h*(img_idx+1)] = zero123_img_90shift
        zero123_img_canvas[2*img_h:3*img_h, img_h*img_idx:img_h*(img_idx+1)] = algo_img

        score_1 = compute_dreamsim(img_ref_path=algo_img_file, img_0_path=zero123_img_file)
        score_2 = compute_dreamsim(img_ref_path=algo_img_file, img_0_path=zero123_90shift_img_file)
        dreamsim_scores.append((score_1, score_2))

    data_dict = {
        'vis_image_paths_zero123': vis_image_paths_zero123,
        'vis_image_paths_zero123_90_shift': vis_image_paths_zero123_90,
        'vis_image_paths_algorithm': vis_image_paths_algorithm,
        'dreamsim_scores': dreamsim_scores
    }

    save_dir = os.path.join(zero123_generation_path, "structural_constistency_outputs", "data.npy")
    if not os.path.exists(os.path.join(zero123_generation_path, "structural_constistency_outputs")):
        os.makedirs(os.path.join(zero123_generation_path, "structural_constistency_outputs"), exist_ok=True)
    np.save(save_dir, data_dict)
    plt.imsave(save_dir.replace('data.npy', 'visualization.png'), zero123_img_canvas)
    
    # greater the dreamsim score, smaller the similarity
    dreamsim_scores = np.stack(dreamsim_scores)
    aggregated_dreamsim_scores = min(dreamsim_scores[:,0].mean(), dreamsim_scores[:,1].mean())
    all_opacity = []
    for opacity_file_path in all_opacity_data:
        opacity = np.asarray(Image.open(opacity_file_path))
        opacity = cv2.resize(opacity, (512, 512))
        opacity = (opacity[...,0]>200)*1.0
        all_opacity.append(opacity.sum()/np.prod(opacity.shape))

    # Reconstructions with very small (smaller than opacity_threshold) rendered mask / opacity are considered 100% faulty.
    mean_opacity = np.stack(all_opacity).mean()
    aggregated_dreamsim_scores = 100. if mean_opacity < opacity_threshold else 100*aggregated_dreamsim_scores
    structural_consistency_metric = 100 - aggregated_dreamsim_scores
    with open(os.path.join(zero123_generation_path, "structural_constistency_outputs", "structural_consistency_metric.txt"), "w") as f:
        f.write("Structural Consistency Metric: {}".format(structural_consistency_metric))
    f.close()
    return structural_consistency_metric

def generate_zero123_multiview_data(rgb_images_path, prompt_id, algorithm_name, uid, tmux_id):
    threestudio_relative_path = "../../Generate3D/threestudio/"
    image_path_0 = os.path.join(rgb_images_path, "0000_cropped_rgba.png")
    image_path_90 = os.path.join(rgb_images_path, "0030_cropped_rgba.png")
    
    command = '''cd {}; python launch.py --config custom/threestudio-mvimg-gen/configs/stable-zero123.yaml --train --gpu 0 data.image_path="{}" object_name="{}" algorithm_name={} --log_job_finish={}'''.format(threestudio_relative_path, image_path_0, prompt_id, algorithm_name, uid)
    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id), command])    

    command = '''cd {}; python launch.py --config custom/threestudio-mvimg-gen/configs/stable-zero123.yaml --train --gpu 0 data.image_path="{}" object_name="{}" algorithm_name={} --log_job_finish={}'''.format(threestudio_relative_path, image_path_90, prompt_id, algorithm_name, uid)
    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id+1), command])    

    while True:
        if os.path.exists(os.path.join(rgb_images_path, "0000_cropped_rgba_{}.txt".format(uid))) and \
            os.path.exists(os.path.join(rgb_images_path, "0030_cropped_rgba_{}.txt".format(uid))):
            subprocess.run(["tmux", "kill-session", "-t", str(tmux_id)])
            subprocess.run(["tmux", "kill-session", "-t", str(tmux_id+1)])
            return True

def evaluate_structural_consistency_metric(prompt_id, algorithm_name, prompt_data_path, tmux_id):
    preprocess_data(prompt_data_path)
    _ = generate_zero123_multiview_data(
        glob.glob(os.path.join(prompt_data_path, "save/it*-test/rgb_images/"))[0], 
        prompt_id, algorithm_name, str(uuid.uuid4()), tmux_id)

    structural_consistency_metric = compute_metric(
        reconstructed_asset_path=prompt_data_path,
        zero123_generation_path=prompt_data_path.replace("/{}/".format(algorithm_name), "/mvimg-gen-zero123-sai/{}/".format(algorithm_name))
    )
    
    return structural_consistency_metric


def main(args):
    if args.prompt_id is not None:
        prompt_data_path = os.path.join(args.algorithm_data_path, args.prompt_id)
        print(f"Using prompt data from: {prompt_data_path}")
        structural_consistency_metric = evaluate_structural_consistency_metric(args.prompt_id, args.algorithm_name, prompt_data_path, args.tmux_id)
        print(f"Prompt data from: {prompt_data_path} has Structural Consistency Value = {structural_consistency_metric}%")
        
    else:
        print(f"Using algorithm data from: {args.algorithm_data_path}")
        for prompt_id in tqdm.tqdm(sorted(os.listdir(args.algorithm_data_path))):
            if len(glob.glob(os.path.join(args.algorithm_data_path, prompt_id, "save/it*-test/rgb_images/*.png")))>0:
                prompt_data_path = os.path.join(args.algorithm_data_path, prompt_id)
                print(f"Using prompt data from: {prompt_data_path}")
                structural_consistency_metric = evaluate_structural_consistency_metric(prompt_id, args.algorithm_name, prompt_data_path, args.tmux_id)
                print(f"Prompt data from: {prompt_data_path} has Structural Consistency Value = {structural_consistency_metric}%")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

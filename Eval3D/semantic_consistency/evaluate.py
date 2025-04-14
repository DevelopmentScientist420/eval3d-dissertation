import argparse
import os, tqdm, subprocess, glob, torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from utils_2d import ViTExtractor, dino_args, dino_vis_pca_mask
from utils_3d import load_mesh, create_renderer, render_mesh, save_rendered_colored_mesh, compute_dino_3d_consistency

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prompt_id', type=str, default=None, help='Path to the prompt data.')
    parser.add_argument('--base_data_path', required=True, type=str)
    parser.add_argument('--mesh_path', default=None, type=str)
    parser.add_argument('--algorithm_name', type=str, default=None, help='')
    parser.add_argument('--tmux_id', type=int, default=1, help='')
    args = parser.parse_args()
    args.algorithm_data_path = os.path.join(args.base_data_path, args.algorithm_name)
    return args

def get_prompt_data(prompt_data_path):
    all_opacity_data = sorted(glob.glob(os.path.join(prompt_data_path, "opacity", "*.png")))
    all_rgb_data = sorted(glob.glob(os.path.join(prompt_data_path, "rgb_images", "*.png")))
    rgb_data = []
    for img in all_rgb_data:
        if 'rgba' in img: continue
        rgb_data.append(img)
    all_rgb_data = rgb_data
    return all_opacity_data, all_rgb_data


def extract_dino_data(prompt_data_path):
    all_opacity_data, all_rgb_data = get_prompt_data(prompt_data_path)
      
    dino_extractor = ViTExtractor(dino_args.model_type, dino_args.stride, device=dino_args.device)
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(dino_args.device)
    upsampler.model = upsampler.model.to(dino_args.device)
    
    selected_opacity_map = []
    selected_upsampled_dino_descriptors = []
    for idx in tqdm.tqdm(range(len(all_rgb_data))):
        if idx%2!=0: continue
        
        rgb_image = np.asarray(Image.open(all_rgb_data[idx]))
        opacity_map = np.array(Image.open(all_opacity_data[idx]))[...,0] / 255.
        rgb_image = cv2.resize(rgb_image, (512, 512))
        opacity_map = cv2.resize(opacity_map, (512, 512))
        opacity_map = (opacity_map>0)*1.

        with torch.no_grad():
            image_batch, image_pil = dino_extractor.preprocess(all_rgb_data[idx], dino_args.load_size, dino_args.patch_size)
            image_batch = image_batch.to(dino_args.device)
            hr_descriptor_feats = upsampler(image_batch)
            hr_descriptor_feats = hr_descriptor_feats.reshape(hr_descriptor_feats.shape[0], hr_descriptor_feats.shape[1], -1)
            hr_descriptor_feats = hr_descriptor_feats.permute(0,2,1)[None]

        selected_opacity_map.append(torch.from_numpy(opacity_map).to(dino_args.device))
        selected_upsampled_dino_descriptors.append(hr_descriptor_feats[0,0])
        
    print('extracting pca....')
    pca_dino_feature_list = dino_vis_pca_mask(selected_upsampled_dino_descriptors, selected_opacity_map, normalize_pca_output=True)
    feat_idx = 0
    if not os.path.exists(os.path.join(prompt_data_path.replace("'", "\\'"), "all_pca_dino_feats")):
        os.system("mkdir -p " + os.path.join(prompt_data_path.replace("'", "\\'"), "all_dino_feats"))
        os.system("mkdir -p " + os.path.join(prompt_data_path.replace("'", "\\'"), "all_pca_dino_feats"))
    
    all_pca_dino_feats = []
    for idx in range(len(all_rgb_data)):
        if idx%2!=0: continue
        dino_upsampled_feat = selected_upsampled_dino_descriptors[feat_idx]
        dino_upsampled_pca_feat = pca_dino_feature_list[feat_idx]
            
        feat_idx += 1

        np.save(os.path.join(prompt_data_path, "all_dino_feats", "{}.npy".format(str(idx).zfill(4))), dino_upsampled_feat.cpu().numpy().reshape(256, 256, -1))
        np.save(os.path.join(prompt_data_path, "all_pca_dino_feats", "{}.npy".format(str(idx).zfill(4))), dino_upsampled_pca_feat.reshape(256, 256, -1))
        all_pca_dino_feats.append(os.path.join(prompt_data_path, "all_pca_dino_feats", "{}.npy".format(str(idx).zfill(4))))

        # print(dino_upsampled_pca_feat.shape, dino_upsampled_pca_feat.min(), dino_upsampled_pca_feat.max(), "dino_upsampled_pca_feat shape min max")
        dino_upsampled_pca_feat = (dino_upsampled_pca_feat - dino_upsampled_pca_feat.min(axis=-1)[...,None]) / (dino_upsampled_pca_feat.max(axis=-1)[...,None] - dino_upsampled_pca_feat.min(axis=-1)[...,None])
        Image.fromarray(np.asarray(dino_upsampled_pca_feat.reshape(256, 256, -1)[...,1:4]*255, dtype=np.uint8)).save(os.path.join(prompt_data_path, 'all_pca_dino_feats', '{}.png'.format(str(idx).zfill(4))))
        
    # os.system('''touch {}'''.format(os.path.join(prompt_data_path.replace("'", "\\'"), "all_dino_feats", "dino_extracted.txt")))
    print("Dino data extracted at: {}".format(os.path.join(prompt_data_path.replace("'", "\\'"), "all_dino_feats")))    
    torch.cuda.empty_cache()
    return all_pca_dino_feats, all_opacity_data
    

def extract_mesh(prompt, algorithm_name, prompt_data_path, tmux_id):
    if len(glob.glob(os.path.join(prompt_data_path.split('@')[0] + "@*", "save/it*-export", "model.obj")))>0:
        # for file in glob.glob(os.path.join(prompt_data_path.split('@')[0] + "@*", "save/it*-export", "model.obj")):
        #     os.system("rm -r -v " + file.split("save")[0])
        return glob.glob(os.path.join(prompt_data_path.split('@')[0] + "@*", "save/it*-export", "model.obj"))[-1]
        
    threestudio_relative_path = "../../Generate3D/threestudio/"
    if "dreamfusion" in algorithm_name or "textmesh" in algorithm_name:
        command = '''cd {}; python launch.py --config configs/{}.yaml --export --gpu 0 system.prompt_processor.prompt="{}" resume="{}/ckpts/last.ckpt" system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256 system.geometry.isosurface_threshold=1. system.exporter.context_type=cuda'''.format(threestudio_relative_path, algorithm_name, prompt, prompt_data_path, prompt_data_path)
    
    elif "mvdream" in algorithm_name:
        command = '''cd {}; python launch.py --config configs/{}.yaml --export --gpu 0 system.prompt_processor.prompt="{}" resume="{}/ckpts/last.ckpt" system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256 system.geometry.isosurface_threshold=1. system.exporter.context_type=cuda'''.format(threestudio_relative_path, algorithm_name, prompt, prompt_data_path, prompt_data_path)

    elif "prolificdreamer" in algorithm_name or "magic3d" in algorithm_name:
        command = '''cd {}; python launch.py --config configs/{}.yaml --export --gpu 0 system.prompt_processor.prompt="{}" resume="{}/ckpts/last.ckpt" system.geometry_convert_from="{}/ckpts/last.ckpt" system.exporter_type=mesh-exporter system.renderer.context_type=cuda system.exporter.context_type=cuda'''.format(threestudio_relative_path, algorithm_name, prompt, prompt_data_path, prompt_data_path)
    
    ### gs
    ## command = '''cd threestudio; CUDA_VISIBLE_DEVICES={} python launch_original.py --config {}.yaml --export --gpu 0 system.prompt_processor.prompt="{}" resume="{}/ckpts/last.ckpt"'''.format(gpu_id, args.config_name, prompt, data_path)

    command = command + "; sleep infinity"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", str(tmux_id), command])

    while True:
        if len(glob.glob(os.path.join(prompt_data_path.split('@')[0] + "@*", "save/it*-export", "model.obj")))>0:
            print("Mesh generated at: {}".format(glob.glob(os.path.join(prompt_data_path.split('@')[0] + "@*", "save/it*-export", "model.obj"))[-1]))
            return glob.glob(os.path.join(prompt_data_path.split('@')[0] + "@*", "save/it*-export", "model.obj"))[-1]


def extract_dino_variance_data(all_batch_data, all_pca_dino_feats, mesh, save_dir, device="cuda:0"):
    
    all_rendered_dino_verts = []
    all_rendered_verts_visibility = []
    feat_idx = 0
    print('Rendering Mesh: ', len(all_batch_data), len(all_pca_dino_feats))
    for idx in tqdm.tqdm(range(len(all_batch_data))):
        if idx%2!=0: continue

        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        
        camera_position = batch_data['camera_positions'].cpu().numpy()
        proj_matrix = batch_data['proj_mtx'].cpu().numpy()
        renderer, rasterizer, cameras = create_renderer(camera_position, proj_matrix, device=device, elevation=batch_data['elevation'], azimuth=batch_data['azimuth'], camera_distances=batch_data['camera_distances'])
        rendered_images, rendered_verts_screen, rendered_verts_visibility = render_mesh(renderer, rasterizer, cameras, mesh)

        descriptors = np.load(all_pca_dino_feats[feat_idx])
        descriptors = descriptors.reshape(-1, 4)
        descriptors_feat_dim = descriptors.shape[1]

        if not torch.is_tensor(descriptors): descriptors = torch.from_numpy(descriptors)
        
        rendered_verts_screen = (rendered_verts_screen - 256) / 256
        rendered_verts_dino = F.grid_sample(
            descriptors.permute(1, 0)[None].reshape(1, descriptors_feat_dim, 256, 256).to(device).float(), 
            grid=rendered_verts_screen[...,:2][:,None])
        rendered_verts_dino = rendered_verts_dino[0,:,0].permute(1,0)
        
        outside_1 = ((rendered_verts_screen[0,...,0]<-1) |  (rendered_verts_screen[0,...,0]>1))
        outside_2 = ((rendered_verts_screen[0,...,1]<-1) |  (rendered_verts_screen[0,...,1]>1))
        outside = (outside_1 | outside_2)
        rendered_verts_visibility[outside] = 0.
        
        all_rendered_dino_verts.append(rendered_verts_dino)
        all_rendered_verts_visibility.append(rendered_verts_visibility)

        feat_idx += 1
    
    # cleaned_mesh, normalized_cleaned_dino_verts_std, dino_verts_mean, dino_verts_std, dino_verts_variance = compute_dino_3d_consistency(mesh, all_rendered_dino_verts, all_rendered_verts_visibility)
    dino_verts_mean, dino_verts_std, dino_verts_variance = compute_dino_3d_consistency(mesh, all_rendered_dino_verts, all_rendered_verts_visibility)
    save_rendered_colored_mesh(mesh, dino_verts_std.cpu().numpy()[...,None], all_batch_data, save_dir, device)
    
    print(dino_verts_std.min(), dino_verts_std.max(), dino_verts_std.shape, "dino_verts_std min max")
    
    if not os.path.exists(os.path.join(save_dir)):
        os.system('mkdir -p ' + os.path.join(save_dir))
    np.save(os.path.join(save_dir, 'dino_verts_mean.npy'), dino_verts_mean.cpu().numpy())
    np.save(os.path.join(save_dir, 'dino_verts_std.npy'), dino_verts_std.cpu().numpy())
    np.save(os.path.join(save_dir, 'all_rendered_verts_visibility.npy'), torch.stack(all_rendered_verts_visibility).cpu().numpy())
    print('dino variance data extracted...')

    return dino_verts_std.cpu().numpy(), torch.stack(all_rendered_verts_visibility).cpu().numpy().sum(axis=0)


def compute_metric(dino_verts_std, visibility_data, all_opacity_data, opacity_threshold=0.01, visibility_threshold=5, std_threshold=0.075):
    # The metric computes % of vertices which have std smaller than std_threshold value.
    # consider vertices visibile in minimum visibility_threshold views
    dino_verts_std = dino_verts_std[visibility_data>=visibility_threshold]

    # considering vertices with std greater than a certain threshold -- to make sure a large reconstructed mesh with small std doesn't get punished. 
    mask = (dino_verts_std>std_threshold)*1.0
    dino_verts_std_invalid_verts = np.sum(mask) / (dino_verts_std.shape[0]+1.e-8)

    all_opacity = []
    for opacity_file_path in all_opacity_data:
        opacity = np.asarray(Image.open(opacity_file_path))
        opacity = cv2.resize(opacity, (512, 512))
        opacity = (opacity[...,0]>200)*1.0
        all_opacity.append(opacity.sum()/np.prod(opacity.shape))

    # Reconstructions with very small (smaller than opacity_threshold) rendered mask / opacity are considered 100% faulty.
    mean_opacity = np.stack(all_opacity).mean()
    dino_verts_std_invalid_verts = 100. if mean_opacity < opacity_threshold else 100*dino_verts_std_invalid_verts
    semantic_consistency_metric = 100 - dino_verts_std_invalid_verts
    return semantic_consistency_metric
    

def evaluate_semantic_consistency_metric(prompt_id, algorithm_name, prompt_data_path, tmux_id, device="cuda:0"):
    save_dir = os.path.join(prompt_data_path, "semantic_consistency_outputs") #.replace("'", "\\'")
    if os.path.exists(os.path.join(save_dir, "semantic_consistency_metric.txt")):
        return open(os.path.join(save_dir, "semantic_consistency_metric.txt"), "r").readlines()
    mesh_path = extract_mesh(prompt_id.split("@")[0].replace("_", " "), algorithm_name, prompt_data_path, tmux_id)
    all_pca_dino_feats, all_opacity_data = extract_dino_data(glob.glob(os.path.join(prompt_data_path, "save/it*-test"))[0])
    
    all_batch_data = sorted(glob.glob(os.path.join(glob.glob(os.path.join(prompt_data_path, "save/it*-test"))[0], "batch_data", "*.npy")))
    mesh = load_mesh(mesh_path, device=device)
    dino_verts_std, visibility_data = extract_dino_variance_data(all_batch_data, all_pca_dino_feats, mesh, save_dir=save_dir, device=device)
    semantic_consistency_metric = compute_metric(dino_verts_std, visibility_data, all_opacity_data)
    subprocess.run(["tmux", "kill-session", "-t", str(tmux_id)])
    with open(os.path.join(save_dir, "semantic_consistency_metric.txt"), "w") as f:
        f.write("Semantic Consistency Metric: {}".format(semantic_consistency_metric))
    f.close()
    return semantic_consistency_metric


def main(args):
    if args.prompt_id is not None:
        prompt_data_path = os.path.join(args.algorithm_data_path, args.prompt_id)
        print(f"Using prompt data from: {prompt_data_path}")
        semantic_consistency_metric = evaluate_semantic_consistency_metric(args.prompt_id, args.algorithm_name, prompt_data_path, args.tmux_id)
        print(f"Prompt data from: {prompt_data_path} has Semantic Consistency Value = {semantic_consistency_metric}%")
        
    else:
        print(f"Using algorithm data from: {args.algorithm_data_path}")
        for prompt_id in tqdm.tqdm(sorted(os.listdir(args.algorithm_data_path))):
            if len(glob.glob(os.path.join(args.algorithm_data_path, prompt_id, "save/it*-test/rgb_images/*.png")))>0:
                prompt_data_path = os.path.join(args.algorithm_data_path, prompt_id)
                print(f"Using prompt data from: {prompt_data_path}")
                semantic_consistency_metric = evaluate_semantic_consistency_metric(prompt_id, args.algorithm_name, os.path.join(args.algorithm_data_path, prompt_id), args.tmux_id)
                print(f"Prompt data from: {prompt_data_path} has Semantic Consistency Value = {semantic_consistency_metric}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
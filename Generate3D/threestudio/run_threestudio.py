import argparse
import json
import os
import csv
import glob

def get_save_path(csv_file_path, prompt_idx):

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)
            if int(row['prompt_idx']) == prompt_idx:
                return row['save path']

    return 'None'


def execute(main_launch_tag):
    launch_tag = '''{}'''.format(main_launch_tag)
    os.system('pwd')
    print(launch_tag)
    os.system(launch_tag)


# # # upgraded diffusers from diffusers==0.19.3 to diffusers==0.24.0
# def execute_dreamcraft3d_texture(prompt, gpu_id, prompt_idx, stage_1_csv_file_path='/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/threestudio_logs_dreamcraft3d-geometry.csv'):
#     print(f"Executing dreamcraft3d-texture function with prompt: {prompt}")
#     image_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/images_1/{}_rgba.png".format(str(prompt_idx).zfill(4))
#     base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
#     resume_from = get_save_path(stage_1_csv_file_path, prompt_idx=prompt_idx)
#     resume_from = os.path.join(base_path, resume_from, 'ckpts', 'last.ckpt')
#     assert(os.path.exists(resume_from))
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-dreamcraft3D/configs/dreamcraft3d-texture.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}" --prompt_idx {} system.geometry_convert_from="{}"'''.format(gpu_id, image_path, prompt, prompt_idx, resume_from)
#     execute(main_launch_tag)

# def execute_dreamcraft3d_geometry(prompt, gpu_id, prompt_idx, stage_1_csv_file_path='/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/threestudio_logs_dreamcraft3d-coarse-neus.csv'):
#     print(f"Executing dreamcraft3d-geometry function with prompt: {prompt}")
#     image_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/images_1/{}_rgba.png".format(str(prompt_idx).zfill(4))
#     base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
#     resume_from = get_save_path(stage_1_csv_file_path, prompt_idx=prompt_idx)
#     resume_from = os.path.join(base_path, resume_from, 'ckpts', 'last.ckpt')
#     assert(os.path.exists(resume_from))
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-dreamcraft3D/configs/dreamcraft3d-geometry.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}" --prompt_idx {} system.geometry_convert_from="{}"'''.format(gpu_id, image_path, prompt, prompt_idx, resume_from)
#     execute(main_launch_tag)

# def execute_dreamcraft3d_coarse_neus(prompt, gpu_id, prompt_idx, stage_1_csv_file_path='/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/threestudio_logs_dreamcraft3d-coarse-nerf.csv'):
#     print(f"Executing dreamcraft3d_coarse_neus function with prompt: {prompt}")
#     image_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/images_1/{}_rgba.png".format(str(prompt_idx).zfill(4))
#     base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
#     resume_from = get_save_path(stage_1_csv_file_path, prompt_idx=prompt_idx)
#     resume_from = os.path.join(base_path, resume_from, 'ckpts', 'last.ckpt')
#     assert(os.path.exists(resume_from))
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-dreamcraft3D/configs/dreamcraft3d-coarse-neus.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}" --prompt_idx {} system.weights="{}"'''.format(gpu_id, image_path, prompt, prompt_idx, resume_from)
#     execute(main_launch_tag)

# def execute_dreamcraft3d_coarse_nerf(prompt, gpu_id, prompt_idx, stage_1_csv_file_path='/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/logs/threestudio_logs_dreamcraft3d-coarse-nerf.csv'):
#     print(f"Executing dreamcraft3d_coarse_nerf function with prompt: {prompt}")
#     image_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/images_1/{}_rgba.png".format(str(prompt_idx).zfill(4))
#     base_path = "/data/vision/torralba/scratch/sduggal/threed_eval/threestudio/"
#     resume_from = get_save_path(stage_1_csv_file_path, prompt_idx=prompt_idx)
#     resume_from = os.path.join(base_path, resume_from, 'ckpts', 'last.ckpt')
#     if os.path.exists(resume_from):
#         main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-dreamcraft3D/configs/dreamcraft3d-coarse-nerf.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}" --prompt_idx {} resume="{}"'''.format(gpu_id, image_path, prompt, prompt_idx, resume_from)
#     else:
#         main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-dreamcraft3D/configs/dreamcraft3d-coarse-nerf.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}" --prompt_idx {}'''.format(gpu_id, image_path, prompt, prompt_idx)
#     execute(main_launch_tag)



# def execute_magic123_refine_sd(prompt, gpu_id, resume_from, image_path):
    # print(f"Executing magic123 refine sd function with prompt: {prompt}")
    # assert(resume_from is not None)
    # resume_from = os.path.join(resume_from, 'ckpts', 'last.ckpt')
    # assert(os.path.exists(resume_from))
    # main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/magic123-refine-sd.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}" system.geometry_convert_from="{}" system.renderer.context_type=cuda'''.format(gpu_id, image_path, prompt, resume_from)
    # execute(main_launch_tag)


# def execute_magic123_coarse_sd(prompt, gpu_id, image_path):
#     print(f"Executing magic123 coarse sd function with prompt: {prompt}")
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/magic123-coarse-sd.yaml --train --gpu 0 data.image_path="{}" system.prompt_processor.prompt="{}"'''.format(gpu_id, image_path, prompt)
#     execute(main_launch_tag)


def execute_prolificdreamer_texture(prompt, gpu_id, resume_from):
    print(f"Executing prolificdreamer-texture function with prompt: {prompt}")
    assert(resume_from is not None)
    resume_from = os.path.join(resume_from, 'ckpts', 'last.ckpt')
    assert(os.path.exists(resume_from))
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/prolificdreamer-texture.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.geometry_convert_from="{}" system.renderer.context_type=cuda'''.format(gpu_id, prompt, resume_from)
    execute(main_launch_tag)

def execute_prolificdreamer_geometry(prompt, gpu_id, resume_from):
    print(f"Executing prolificdreamer function with prompt: {prompt}")
    assert(resume_from is not None)
    resume_from = os.path.join(resume_from, 'ckpts', 'last.ckpt')
    assert(os.path.exists(resume_from))
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/prolificdreamer-geometry.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.geometry_convert_from="{}" system.renderer.context_type=cuda'''.format(gpu_id, prompt, resume_from)
    execute(main_launch_tag)

def execute_prolificdreamer(prompt, gpu_id):
    print(f"Executing prolificdreamer function with prompt: {prompt}")
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/prolificdreamer.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.loss.lambda_sparsity=50.'''.format(gpu_id, prompt)
    execute(main_launch_tag)


def execute_magic3d_refine_sd(prompt, gpu_id, resume_from):
    print(f"Executing magic123_refine_sd function with prompt: {prompt}")
    assert(resume_from is not None)
    resume_from = os.path.join(resume_from, 'ckpts', 'last.ckpt')
    assert(os.path.exists(resume_from))
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.geometry_convert_from="{}" system.renderer.context_type=cuda'''.format(gpu_id, prompt, resume_from)
    execute(main_launch_tag)


def execute_magic3d_coarse_if(prompt, gpu_id):
    print(f"Executing magic123_coarse_if function with prompt: {prompt}")
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/magic3d-coarse-if.yaml --train --gpu 0 system.prompt_processor.prompt="{}"'''.format(gpu_id, prompt)
    execute(main_launch_tag)

def execute_textmesh_if(prompt, gpu_id):
    print(f"Executing latentnerf function with prompt: {prompt}")
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/textmesh-if.yaml --train --gpu 0 system.prompt_processor.prompt="{}"'''.format(gpu_id, prompt)
    execute(main_launch_tag)


# def execute_latentnerf_refine(prompt, gpu_id, resume_from):
#     assert(os.path.exists(resume_from))
#     print(f"Executing latentnerf refine function with prompt: {prompt}")
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/latentnerf-refine.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.weights="{}"'''.format(gpu_id, prompt, resume_from)
#     execute(main_launch_tag)

# def execute_latentnerf(prompt, gpu_id):
#     print(f"Executing latentnerf function with prompt: {prompt}")
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/latentnerf.yaml --train --gpu 0 system.prompt_processor.prompt="{}"'''.format(gpu_id, prompt)
#     execute(main_launch_tag)

# def execute_threestudio_3dgs(prompt, gpu_id):
#     print(f"Executing threestudio_3dgs function with prompt: {prompt}")
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_shading.yaml --train --gpu 0 system.prompt_processor.prompt="{}"'''.format(gpu_id, prompt)
#     execute(main_launch_tag)

# def execute_threestudio_3dgs_shape(prompt, gpu_id):
#     print(f"Executing threestudio_3dgs function with prompt: {prompt}")
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.geometry.geometry_convert_from="shap-e:{}"'''.format(gpu_id, prompt, prompt)
#     execute(main_launch_tag)

# def execute_mvdream(prompt, gpu_id):
#     print(f"Executing mvdream function with prompt: {prompt}")
#     main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config custom/threestudio-mvdream/configs/mvdream-sd21-shading.yaml --train --gpu 0 system.prompt_processor.prompt="{}"'''.format(gpu_id, prompt)
#     execute(main_launch_tag)

def execute_dreamfusion_if(prompt, gpu_id):
    print(f"Executing dreamfusion-if function with prompt: {prompt}")
    main_launch_tag = '''CUDA_VISIBLE_DEVICES={} python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="{}" system.background.random_aug=true'''.format(gpu_id, prompt)
    execute(main_launch_tag)

def main():
    parser = argparse.ArgumentParser(description="Execute a function based on a selected prompt from a JSON file.")
    parser.add_argument("--algorithm_name", required=True, help="Name of the algorithm in the JSON file.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_idx", type=int, default=None, help="Index of the prompt to select from the JSON file.")
    parser.add_argument("--gpu_id", type=int, required=True, help="gpu_id.")
    parser.add_argument("--resume_from", type=str, default=None, help="required for multi-stage models")
    args = parser.parse_args()

    assert(args.prompt is None or args.prompt_idx is None)

    if args.prompt_idx is not None:
        prompts_library = "../../data/prompt_library.json"
        with open(prompts_library, 'r') as file:
            data = json.load(file)
            prompts = data['dreamfusion']

        selected_prompt_ids_path = "../../data/selected_prompt_ids.json"
        with open(selected_prompt_ids_path, 'r') as file:
            selected_prompt_ids = json.load(file)

        if not (0 <= args.prompt_idx < len(prompts)):
            print(f"Error: Prompt index '{args.prompt_idx}' is out of range for the selected algorithm.")
            return

        if not (0 <= args.prompt_idx < len(selected_prompt_ids)):
            print(f"Error: Prompt index '{args.prompt_idx}' is out of range for the selected algorithm.")
            assert(False)

        selected_prompt = prompts[selected_prompt_ids[args.prompt_idx]]

    if args.prompt is not None: selected_prompt = args.prompt
    
    if args.algorithm_name == 'dreamfusion_if':
        execute_dreamfusion_if(selected_prompt, args.gpu_id)
    elif args.algorithm_name == 'prolificdreamer':
        execute_prolificdreamer(selected_prompt, args.gpu_id)
    elif args.algorithm_name == 'prolificdreamer_geometry':
        execute_prolificdreamer_geometry(selected_prompt, args.gpu_id, args.resume_from)
    elif args.algorithm_name == 'prolificdreamer_texture':
        execute_prolificdreamer_texture(selected_prompt, args.gpu_id, args.resume_from)
    elif args.algorithm_name == 'textmesh_if':
        execute_textmesh_if(selected_prompt, args.gpu_id)
    elif args.algorithm_name == 'magic3d_coarse_if':
        execute_magic3d_coarse_if(selected_prompt, args.gpu_id)
    elif args.algorithm_name == 'magic3d_refine_sd':
        execute_magic3d_refine_sd(selected_prompt, args.gpu_id, args.resume_from)
    
    # elif args.algorithm_name == 'magic123_coarse_sd':
    #     execute_magic123_coarse_sd(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'magic123_refine_sd':
    #     execute_magic123_refine_sd(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'dreamcraft3d_coarse_nerf':
    #     execute_dreamcraft3d_coarse_nerf(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'dreamcraft3d_coarse_neus':
    #     execute_dreamcraft3d_coarse_neus(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'dreamcraft3d_geometry':
    #     execute_dreamcraft3d_geometry(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'dreamcraft3d_texture':
    #     execute_dreamcraft3d_texture(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'mvdream':
    #     execute_mvdream(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'threestudio_3dgs':
    #     execute_threestudio_3dgs(selected_prompt, args.gpu_id)
    # elif args.algorithm_name == 'threestudio_3dgs_shape':
    #     execute_threestudio_3dgs_shape(selected_prompt, args.gpu_id)
    else:
        assert(True==False)

if __name__ == "__main__":
    main()

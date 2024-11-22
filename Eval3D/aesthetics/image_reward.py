import os
import torch
import ImageReward as RM
import glob
import json
from tqdm import tqdm


model = RM.load("ImageReward-v1.0")


def get_score(prompt, img_list):
    with torch.no_grad():
        ranking, rewards = model.inference_rank(prompt, img_list)
        
        return rewards
    
    
def get_imagereward_for_dir(image_directory):
    if os.path.exists(os.path.join(image_directory, "image_reward.json")):
        return
    
    images = glob.glob(os.path.join(image_directory, "*.png"))
    
    prompt = os.path.basename(image_directory[:-1]) if image_directory.endswith("/") else os.path.basename(image_directory)
    
    scores = get_score(prompt, images)
    
    score_dict = {f"{os.path.basename(image)}": score for image, score in zip(images, scores)}
    
    with open(os.path.join(image_directory, "image_reward.json"), "w") as f:
        json.dump(score_dict, f, indent=4)
        
    return
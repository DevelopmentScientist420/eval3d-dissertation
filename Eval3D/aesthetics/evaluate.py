import json, os, glob, sys, re
from collections import defaultdict
from utils import extract_frames
from image_reward import get_imagereward_for_dir
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Evaluate Aesthetic Score")

def process_directory_imagereward(directory, n_div=60):
    imagereward_score_dict = json.load(open(os.path.join(directory, 'image_reward.json')))
    this_scores = []
    
    for i in range(n_div):
        this_scores.append(imagereward_score_dict[f"{i}.png"])
        
    # now pooling is here
    looped_scores = this_scores + this_scores + this_scores
    pooled_scores = []
    window=1
    for i in range(n_div, 2*n_div):
        pooled_scores.append(np.min(looped_scores[i-window:i+window+1]))
        
    return np.mean(pooled_scores)


def get_aesthetic_scores(video_filename, cache_dir="sample_results", n_div=60):
    frames = extract_frames(video_filename, cache_dir,n_div=60)
    
    # compute scores
    get_imagereward_for_dir(cache_dir, n_div=60)
    
    # get scores
    score = process_directory_imagereward(cache_dir, n_div=60)
    
    # clean up the cache folder
    for f in glob.glob(os.path.join(cache_dir, '*')):
        os.remove(f)
    os.rmdir(cache_dir)
    return score

if __name__ == "__main__":
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--cache_dir', type=str, default='sample_results', help='Directory to cache results')
    args = parser.parse_args()
    
    video_filename = args.video
    cache_dir = args.cache_dir
    n_div = 60
    if not os.path.exists(video_filename):
        print(f"Video file {video_filename} does not exist.")
        sys.exit(1)
    
    # compute aesthetic scores
    score = get_aesthetic_scores(video_filename, cache_dir, n_div)
    print(f"Aesthetic score for {video_filename}: {score}")
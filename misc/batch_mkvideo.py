import imageio
import os
import sys
import cv2
import numpy as np
import shutil


base_dir = './exp_satellite'

collect_dir = './exp_satellite_superclose_video'
os.makedirs(collect_dir, exist_ok=True)


for exp_dir in os.listdir(base_dir):
    render_dir = os.path.join(base_dir, exp_dir, 'render_only_path_superclose_00300000')
    if not os.path.isdir(render_dir):
        print('skipping: ', render_dir)
        continue

    item_list = sorted([item for item in os.listdir(render_dir) if item.endswith('.png') and 'normal' not in item], key=lambda x: int(x[:x.find('.')]))
    frames_rgb = [imageio.imread(os.path.join(render_dir, item))  for item in item_list]

    item_list = sorted([item for item in os.listdir(render_dir) if item.endswith('.png') and 'normal' in item], key=lambda x: int(x[:x.find('_')]))
    frames_normal = [imageio.imread(os.path.join(render_dir, item))  for item in item_list]

    frames = [np.concatenate((rgb, normal), axis=1) for rgb, normal in zip(frames_rgb, frames_normal)]
    frames = frames + frames[::-1]

    imageio.mimwrite(render_dir+'.mp4', frames, fps=30, quality=8)

    shutil.copy2(render_dir+'.mp4', os.path.join(collect_dir, exp_dir+'.mp4'))


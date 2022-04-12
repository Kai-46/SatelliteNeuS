import imageio
import os
import sys
import cv2
import numpy as np

base_dir = sys.argv[1]

item_list = sorted([item for item in os.listdir(base_dir) if item.endswith('.png') and 'normal' not in item], key=lambda x: int(x[:x.find('.')]))
frames_rgb = [imageio.imread(os.path.join(base_dir, item))  for item in item_list]

item_list = sorted([item for item in os.listdir(base_dir) if item.endswith('.png') and 'normal' in item], key=lambda x: int(x[:x.find('_')]))
frames_normal = [imageio.imread(os.path.join(base_dir, item))  for item in item_list]

frames = [np.concatenate((rgb, normal), axis=1) for rgb, normal in zip(frames_rgb, frames_normal)]
frames = frames + frames[::-1]

imageio.mimwrite(base_dir+'.mp4', frames, fps=30, quality=8)

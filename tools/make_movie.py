#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Mar 10 16:00:50 2020

@author: zhaoxm
"""

import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


def mkvideo(im_seq_path, video_file, image_size, fps=12, use_cv=True):
    if use_cv:
        writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'XVID'), fps, image_size)
    else:
        writer = FFMPEG_VideoWriter(video_file, size=image_size, fps=fps)
    for im_path in tqdm(im_seq_path):
        im = cv2.imread(str(im_path))
        if use_cv:
            writer.write(im)
        else:
            im = im[:, :, ::-1]
            writer.write_frame(im)
    if use_cv:
        cv2.destroyAllWindows()
        writer.release()
    else:
        writer.close()

def sh_mkvideo():
    os.system("ffmpeg -r 1 -i %07d.jpg -vcodec mpeg4 -y test.mp4")
    
if __name__ == '__main__':
    path = r'/home/zhxm/datasets/fusion_dataset/trial/trial_2020_02_20_16-45-57#2020-06-10_09:55'
    save_path = path + '.avi'
    
    im_seq = sorted(Path(path).glob('*.jpg'))
    image_size = cv2.imread(str(im_seq[0])).shape[1::-1]
    mkvideo(im_seq, save_path, image_size)
    
#%%
#path = r'/home/zhaoxm/datasets/tmp/fusion_results/fused4557/maps_resize'
#im_seq = sorted(Path(path).glob('*.jpg'))
#save_path = Path(r'/home/zhaoxm/datasets/tmp/fusion_results/fused4557/maps_resize')
#save_path.mkdir(parents=True, exist_ok=True)
#for img_path in tqdm(im_seq):
#    im = cv2.imread(str(img_path))
#    im_ = cv2.resize(im, (1000, 1000), interpolation=cv2.INTER_LINEAR)
#    save_name = save_path / img_path.name
#    cv2.imwrite(str(save_name), im_)
    

    
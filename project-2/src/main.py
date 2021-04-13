import numpy as np
import cv2
import json

import glob 
from pathlib import Path 

import matplotlib.pyplot as plt
import pandas as pd

from orb import *
import re

def get_zf(video_idx):
    folder = Path('input',f'zf{video_idx}',f'ZebraFish-0{video_idx}')

    # front view images
    front_imgs = [img for img in sorted((folder/Path('imgF')).iterdir()) if img.is_file()]
    # top view images
    top_imgs = [img for img in sorted((folder/Path('imgT')).iterdir()) if img.is_file()]
    # annotations  
    annotations = folder / Path('gt', 'gt.txt')
    # aquarium coordinate   
    camT_ref = remove_json_coments(folder/Path('camT_references.json'))
    with open(camT_ref) as f:
        data = json.load(f)
        aquarium_topLeft_top = data[0]['camera']
        aquarium_botRight_top = data[2]['camera']
    camF_ref = remove_json_coments(folder/Path('camF_references.json'))
    with open(camF_ref) as f:
        data = json.load(f)
        aquarium_topLeft_front = data[0]['camera']
        aquarium_botRight_front = data[2]['camera']
        
    aquarium_refs = {
        "top_view":{ 
            "aquarium_topLeft":(int(int(aquarium_topLeft_top['x'])//2), int(int(aquarium_topLeft_top['y'])//2)),
            "aquarium_botRight":(int(int(aquarium_botRight_top['x'])//2), int(int(aquarium_botRight_top['y'])//2))
        },
        "front_view":{  
            "aquarium_topLeft":(int(int(aquarium_topLeft_front['x'])//2), int(int(aquarium_topLeft_front['y'])//2)),
            "aquarium_botRight":(int(int(aquarium_botRight_front['x'])//2), int(int(aquarium_botRight_front['y'])//2))
        }
    }

    zf = {"front_view":front_imgs,
          "top_view":top_imgs,
          "annotation":annotations,
          "refs":aquarium_refs}

    return zf

def remove_json_coments(path):
    
    new_lines = []
    clean_path = Path(path.parents[0], path.stem +'_clean' + path.suffix) 
    with open(path, 'r') as f:
        for line in f.readlines(): 
            new_lines.append(re.split('/\\*.*?\\*/', line)[0])
    with open(clean_path, 'w') as f:
        f.writelines(new_lines) 
    return clean_path

def crop2aquarium(img, zf_view): 
    topLeft = zf_view['aquarium_topLeft']
    botRight = zf_view['aquarium_botRight']
    
    return img[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

def read_annotation(gt):
    columns = [
        'frame',
        'id',
        '3d_x',
        '3d_y',
        '3d_z',
        'camT_x',
        'camT_y',
        'camT_left',
        'camT_top',
        'camT_width',
        'camT_height',
        'camT_occlusion',
        'camF_x',
        'camF_y',
        'camF_left',
        'camF_top',
        'camF_width',
        'camF_height',
        'camF_occlusion'
    ] 
    rows =[]
    with open(gt,'r') as f:
        for line in f.readlines():
            split_line = line.split(',') 
            rows.append(split_line) 
    df = pd.DataFrame(columns=columns, data=rows)
    df = df.drop('camF_occlusion', 1)
    df = df.drop('camT_occlusion', 1)
    df = df.drop('3d_x', 1)
    df = df.drop('3d_y', 1)
    df = df.drop('3d_z', 1)
    df = df.drop('camT_x', 1)
    df = df.drop('camT_y', 1)
    df = df.drop('camF_x', 1)
    df = df.drop('camF_y', 1)
    print(df)

    return df
 

def main():     
    datasets = [2]
    zfs = [get_zf(i) for i in datasets]

    for zf in zfs:

        assert len(zf['front_view']) == len(zf['top_view']), \
            'number of frames per view must match'
  
        gt = read_annotation(zf['annotation']) 

        nfishes = int(gt['id'].max())
        fishes_kp_front = [[]]*nfishes
        fishes_kp_top = [[]]*nfishes

        for frame_n in range(50):
            # Front view frame:
            front_img_path = zf['front_view'][frame_n]
            print(front_img_path)
            front_img = cv2.imread(str(front_img_path), 0) 
            front_img = crop2aquarium(front_img, zf['refs']['front_view'])

            #top view frame
            top_img_path = zf['top_view'][frame_n]
            top_img = cv2.imread(str(top_img_path), 0) 
            top_img = crop2aquarium(top_img, zf['refs']['top_view'])
     
            this_frame_df = gt.loc[gt['frame']==str(frame_n+1)]

            # ORB DETECT
            # FRONT
            orb_front = ORB()
            orb_front.detect_and_extract(front_img)

            # TOP
            orb_top = ORB()
            orb_top.detect_and_extract(top_img)  
            
            # front match with fish
            for i, kp in enumerate(orb_front.keypoints):  
                for fish_id in range(1, nfishes+1):  
                    #select current fish on current frame
                    this_fish = this_frame_df.loc[this_frame_df['id']==str(fish_id)].to_numpy().astype(int)[0] 
                    # frame0 id1 camT_left2 camT_top3 camT_width4 camT_height5 camF_left6 camF_top7 camF_width8 camF_height9
                    in_xbox = this_fish[6]//2 < kp[0] and kp[0] < this_fish[6]//2+this_fish[8]//2
                    in_ybox = front_img.shape[0]-this_fish[7]//2 > kp[1] and kp[1] < front_img.shape[0]-(this_fish[7]//2+this_fish[9]//2)

                    if in_xbox and in_ybox: fishes_kp_front[fish_id-1].append((kp.tolist(), orb_front.descriptors[i]))

            # top match with fish
            for i, kp in enumerate(orb_top.keypoints):  
                for fish_id in range(1, nfishes+1):  
                    #select current fish on current frame
                    this_fish = this_frame_df.loc[this_frame_df['id']==str(fish_id)].to_numpy().astype(int)[0] 
                    # frame0 id1 camT_left2 camT_top3 camT_width4 camT_height5 camF_left6 camF_top7 camF_width8 camF_height9
                    in_xbox = this_fish[2]//2 < kp[0] and kp[0] < this_fish[2]//2+this_fish[4]//2 
                    in_ybox = top_img.shape[0]-this_fish[3]//2 > kp[1] and kp[1] < top_img.shape[0]-(this_fish[3]//2+this_fish[5]//2) 
                    if in_xbox and in_ybox: fishes_kp_top[fish_id-1].append((kp.tolist(), orb_top.descriptors[i]))


            # filtered_kps_front = [[descriptor[0] for descriptor in fish] for fish in fishes_kp_front][0] 
            # color_img = cv2.cvtColor(front_img,cv2.COLOR_GRAY2RGB)
            # for i, kp in enumerate(orb_front.keypoints): 
            #     cv2.circle(color_img, (kp[0],kp[1]), 5, (0,0,255), thickness=1, lineType=8, shift=0)
            # cv2.imshow('front_kp', color_img)
            # cv2.waitKey(0) 

            # filtered_kps_top = [[descriptor[0] for descriptor in fish] for fish in fishes_kp_top][0] 
            # color_imga = cv2.cvtColor(top_img,cv2.COLOR_GRAY2RGB)
            # for i, kp in enumerate(orb_top.keypoints):  
            #     cv2.circle(color_imga, (kp[0],kp[1]), 5, (0,255,0), thickness=1, lineType=8, shift=0)
            # cv2.imshow('top_kp', color_imga)
            # cv2.waitKey(0) 



if __name__ == "__main__":
    main()


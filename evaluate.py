#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# import sys
# import copy
# import cv2
import torch
# print(torch.__version__)
# import scipy
# import requests
# import torch
import json
# import trimesh
# import einops
# import torchvision
import numpy as np
# import pandas as pd
import kaolin as kal
import os.path as osp
import kaolin.ops.mesh
# import matplotlib.pyplot as plt

from mesh import Mesh
from render import Renderer
from tqdm.auto import tqdm
from collections import defaultdict, Counter
# from IPython.display import display
# from torchvision.ops import box_convert
from normalization import MeshNormalizer
# from PIL import Image, ImageDraw, ImageFont
# from huggingface_hub import hf_hub_download
# from pytorch_lightning import seed_everything

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cache_proj', type=str, required=True)
parser.add_argument('--max_elev_angle', type=int, required=True)

args = parser.parse_args()
eval_list = [f'cache{s.split("-")[0]}/sam3D_dino_{s.split("-")[1]}' for s in args.cache_proj.split(',')]


# Now segment FAUST dataset (SATR's version?)
dataset_dir = 'data/MPI-FAUST/training/sampled_scans/'
models = [f for f in os.listdir(dataset_dir) if osp.isfile(osp.join(dataset_dir, f)) and '_seg' not in f and 'obj' in f]





# In[ ]:


# Now evaluate the output and report the numbers
def calculate_shape_IoU(pred:np.array, gt:np.array, label:str):
    # pred: np.array [N_points]
    # seg: np.array [N_points]
    # Adopted from https://github.com/antao97/dgcnn.pytorch/blob/f4da503444ce663b06c8d1bca79e746ef1647b18/main_partseg.py
    I = np.sum(np.logical_and(pred == label, gt == label))
    U = np.sum(np.logical_or(pred == label, gt == label))

    if U == 0:
        iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
    else:
        iou = I / float(U)

    return iou

def calc_IoU_torch(pred, gt, label):
    I = torch.logical_and(pred == label, gt == label).sum()
    U = torch.logical_or(pred == label, gt == label).sum()
    
    if U == 0:
        iou = 1
    else:
        iou = I / float(U)
        
    return iou

def get_face_labels(verts_labels, mesh, labelDict):
    # Get the gt face labels by parsing the vertices labels
    mesh_face_labels = ['' for _ in range(len(mesh.faces))]
    for i_f, face_verts in enumerate(mesh.faces):
        label = Counter([verts_labels[idx] for idx in face_verts]).most_common(1)[0][0]
        mesh_face_labels[i_f] = labelDict[label]
        
    return mesh_face_labels

def get_seg_mask(image_face_ids, mesh_face_labels):
    # Now you know the gt labels for each face in the mesh, you can now generate a matrix of the same size of face_ids matrix but labeled
    image_face_labels = [[] for _ in range(image_face_ids.shape[0])]
    for i_row in range(image_face_ids.shape[0]):
        for i_col in range(image_face_ids.shape[1]):
            face_label = -1 if image_face_ids[i_row][i_col] == -1 else mesh_face_labels[image_face_ids[i_row][i_col]]
            image_face_labels[i_row].append(face_label)

    return np.array(image_face_labels)

# Now evaluate the clipseg performance 

# Read the predicted cls for each mesh
# Read the ground-truth 
dataDir = osp.join('data/MPI-FAUST/training/sampled_scans/')
# import math
# np.set_printoptions(threshold=math.inf)

for eval_path in eval_list:
    partsIous = defaultdict(list)
    partsIoUs2D = defaultdict(list)
    all_image_face_idx = torch.load(osp.join('./saved_renders', f'{args.max_elev_angle}.pt'))

    for i, m in tqdm(enumerate(models)):
        if i > 8:
            break
        mesh_path = os.path.join(dataDir, m)
        mesh_name = m.split('.')[0]
        output_dir = os.path.join(eval_path, mesh_name)
        
        with open(os.path.join(output_dir, f"output.json")) as fin:
            pred = np.array(json.load(fin))
        
        with open(os.path.join(f"final_coarse.json")) as fin:
            gt = np.array(json.load(fin)[f'{mesh_name}'])

        
        gtLabels = sorted(set(gt))
        assert 'unknown' not in gt
        
        for l in gtLabels:
            partsIous[l].append(calculate_shape_IoU(pred, gt, l))
        assert(len(pred) == len(gt))
        
        mesh = Mesh(mesh_path)
        # normalize mesh in unit sphere
        _ = MeshNormalizer(mesh)()
        
        labelDict = defaultdict(lambda:-1, {label:i for i,label in enumerate(gtLabels)})
        # labelDict |= 
        gt_face_labels = get_face_labels(gt, mesh, labelDict)
        pred_face_labels = get_face_labels(pred, mesh, labelDict)
        
        for j in range(120):
            image_face_ids = all_image_face_idx[i][j]
            gt_seg = get_seg_mask(image_face_ids, gt_face_labels)
            pred_seg = get_seg_mask(image_face_ids, pred_face_labels)
            
            for l in gtLabels:
                l = labelDict[l]
                partsIoUs2D[l].append(calculate_shape_IoU(pred_seg, gt_seg, l))
                # gt_upper_left_i, gt_upper_left_j = next((i,j) for i in range(512) for j in range(512) if gt_seg == l)
                # gt_lower_right_i, gt_lower_right_j = next((i,j) for i in reversed(range(512)) for j in reversed(range(512)) if gt_seg == l)
                        
        # else: break
                
        
    # calculate 3D mIoU
    mIous = []
    for k, v in partsIous.items():
        partIoU = np.mean(v)
        print(f'{k} 3D mIoU: {partIoU}')
        mIous.append(partIoU)

    print(f'{eval_path} 3D mIoU = {np.mean(mIous)}\n')
    
    # calculate 2D segmentation mIoU
    mIoU2D = []
    for k,v in partsIoUs2D.items():
        partIoU = np.mean(v)
        print(f'{k} 2D mIoU: {partIoU}')
        mIoU2D.append(partIoU)
        
    print(f'{eval_path} 2D mIoU = {np.mean(mIoU2D)}\n')


# In[ ]:





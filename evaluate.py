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
    mesh_face_labels = [-1 for _ in range(len(mesh.faces))]
    for i_f, face_verts in enumerate(mesh.faces):
        label = Counter([verts_labels[idx] for idx in face_verts]).most_common(1)[0][0]
        mesh_face_labels[i_f] = labelDict[label]
        
    return mesh_face_labels

def get_seg_mask(image_face_ids, mesh_face_labels):
    # Now you know the gt labels for each face in the mesh, you can now generate a matrix of the same size of face_ids matrix but labeled
    # image_face_ids [512, 512]
    # mesh_face_labels []
    # image_face_labels = [ [] for _ in range(image_face_ids.shape[0])]
    image_face_labels = np.ones_like(image_face_ids.cpu().numpy()) * -1
    mesh_face_labels = np.array(mesh_face_labels)
    
    """
    numpy.take(a, indices, axis=None, out=None, mode='raise')[source]
Take elements from an array along an axis.

When axis is not None, this function does the same thing as “fancy” indexing (indexing arrays using arrays); however, it can be easier to use if you need elements along a given axis. A call such as np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].

Explained without fancy indexing, this is equivalent to the following use of ndindex, which sets each of ii, jj, and kk to a tuple of indices:

Ni, Nk = a.shape[:axis], a.shape[axis+1:]
Nj = indices.shape
for ii in ndindex(Ni):
    for jj in ndindex(Nj):
        for kk in ndindex(Nk):
            out[ii + jj + kk] = a[ii + (indices[jj],) + kk]
    """
    # image_face_labels_flattened = image_face_labels.flatten()
    # iamge_face_label_mask = image_face_ids.cpu().numpy().flatten() == -1
    # image_face_ids[image_face_ids.cpu().numpy() == -1] = 0
    # image_face_labels_flattened = np.take(image_face_labels_flattened, image_face_ids.cpu().numpy().flatten())
    # image_face_labels_flattened[iamge_face_label_mask] = -1
    # image_face_labels = image_face_labels_flattened.reshape(image_face_labels.shape)  
    # np.take()
    for i_row in range(image_face_ids.shape[0]):
        for i_col in range(image_face_ids.shape[1]):
            face_label = -1 if image_face_ids[i_row][i_col] == -1 else mesh_face_labels[image_face_ids[i_row][i_col]]
            image_face_labels[i_row][i_col] = face_label

    return np.array(image_face_labels)


def get_detection_iou(gt_x1, gt_y1, gt_x2, gt_y2, pred_x1, pred_y1, pred_x2, pred_y2):
    x_left = max(gt_x1, pred_x1)
    y_top = max(gt_y1, pred_y1)
    x_right = min(gt_x2, pred_x2)
    y_bottom = min(gt_y2, pred_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    I = (x_right - x_left) * (y_bottom - y_top)
    
    gt_area = (gt_x2-gt_x1)*(gt_y2-gt_y1)
    pred_area = (pred_x2-pred_x1)*(pred_y2-pred_y1)
    
    U = float(gt_area+pred_area-I)
    
    IoU = I / U
    assert 0. <= IoU <= 1.
    return IoU

# Now evaluate the clipseg performance 

# Read the predicted cls for each mesh
# Read the ground-truth 
dataDir = osp.join('data/MPI-FAUST/training/sampled_scans/')
# import math
# np.set_printoptions(threshold=math.inf)

for eval_path in eval_list:
    partsIous = defaultdict(list)
    partsIoUs2D = defaultdict(list)
    partsIoUsDetection = defaultdict(list)
    
    all_image_face_idx = torch.load(osp.join('./saved_renders', f'{args.max_elev_angle}.pt'))
    
    if eval_path[-2] == 'a':
        bbox_path = osp.join('./saved_bboxes', f'{eval_path.split("/")[0]}-augmented_images')
    else:
        bbox_path = osp.join('./saved_bboxes', f'{eval_path.split("/")[0]}-rendered_images')
        
    all_bboxes = torch.load(bbox_path)

    for i, m in tqdm(enumerate(models)):
        if i > 0:
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
        
        # for j in range(all_image_face_idx.shape[1]):
        for j in range(10):
            image_face_ids = all_image_face_idx[i][j]
            gt_seg = get_seg_mask(image_face_ids, gt_face_labels)
            pred_seg = get_seg_mask(image_face_ids, pred_face_labels)
            
            bboxes = all_bboxes[i][j]
            
            for l in gtLabels:
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow((gt_seg == labelDict[l]))
                # plt.savefig('test_figure_seg.png')
                # from PIL import Image
                # a = for row in reversed(range(512))
                try:
                    # all_pos = [(row,col) for row in range(512) for col in range(512) \
                    #                           if gt_seg[row][col] == labelDict[l]]
                    gt_min_y, _ = next((row,col) for row in range(512) for col in range(512) \
                                              if gt_seg[row][col] == labelDict[l])
                    gt_max_y, _ = next((row,col) for row in reversed(range(512)) for col in reversed(range(512)) \
                                              if gt_seg[row][col] == labelDict[l])
                    _, gt_min_x = next((row,col) for col in range(512) for row in range(512)\
                                              if gt_seg[row][col] == labelDict[l])
                    _, gt_max_x = next((row,col) for col in reversed(range(512)) for row in reversed(range(512))\
                                              if gt_seg[row][col] == labelDict[l])
                except StopIteration:
                    partsIoUsDetection[l].append(0.)
                    print('StopIteration')
                    continue
                
                if bboxes[l] is not None:
                    print(f'bboxes[l]:{l}, {bboxes[l]}')
                    print(gt_min_x, gt_min_y, gt_max_x, gt_max_y, *(bboxes[l]))
                    print(*[el.item() for el in bboxes[l]])
                    print(get_detection_iou(gt_min_x, gt_min_y, gt_max_x, gt_max_y, *[el.item() for el in bboxes[l]]))
                    partsIoUsDetection[l].append(get_detection_iou(gt_min_x, gt_min_y, gt_max_x, gt_max_y, *[el.item() for el in bboxes[l]]))
                else:
                    partsIoUsDetection[l].append(0.)
                    
                l = labelDict[l]
                partsIoUs2D[l].append(calculate_shape_IoU(pred_seg, gt_seg, l))
                    
                
        
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
    
    # calculate 2D detection mIoU
    mIoUDetection = []
    for k,v in partsIoUsDetection.items():
        partIoU = np.mean(v)
        print(f'{k} Detection mIoU: {partIoU}')
        mIoUDetection.append(partIoU)
        
    print(f'{eval_path} Detection mIoU = {np.mean(mIoUDetection)}\n')





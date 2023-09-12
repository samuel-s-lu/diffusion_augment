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
import os
import kaolin.ops.mesh
# import matplotlib.pyplot as plt

from mesh import Mesh
from render import Renderer
from tqdm.auto import tqdm
from collections import defaultdict
# from IPython.display import display
# from torchvision.ops import box_convert
from normalization import MeshNormalizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_elev_angle', type=int, required=True)

args = parser.parse_args()


def sample_viewpoints(): # 120
    elevs = []
    azimuth = []
    
    min_radius = 1.5
    max_radius = 2.0
    
    # 5 different elevations (negatives and positive angles)
    if args.max_elev_angle == 90:
        elevations = list((np.arange(start=0, stop=90, step=30)))
    elif args.max_elev_angle == 75:
        elevations = list((np.arange(start=0, stop=75, step=25)))
    elif args.max_elev_angle == 60:
        elevations = list((np.arange(start=0, stop=60, step=20)))
    else:
        raise RuntimeError('Invalid max elevation angle specified')
    
    n_pos_elevations = len(elevations)
    for i in range(n_pos_elevations):
        elevations.append(-1 * elevations[i])
    elevations = list(set(elevations))
    
    azimuths = list(np.arange(start=0, stop=360, step=45)) # 8 different azimuths
    radii = [max(min_radius, max_radius * np.sin(np.deg2rad(90-e))) for e in elevations[:n_pos_elevations]]
    viewpoints_set = defaultdict(list)

    viewpoints = {}
    n_views = 0
    for i, r in enumerate(radii):
        n_v = 0
        viewpoints[i] = {
            'radius': r,
            'elevations': [],
            'azimuths' : []
        }
        
        for a in azimuths:
            for e in elevations:
                viewpoints[i]['elevations'].append(e)
                viewpoints[i]['azimuths'].append(a)
                n_views += 1
                n_v += 1
        
        viewpoints[i]['n_views'] = n_v
                
    return viewpoints, n_views



def get_face_idx(mesh_path):
    
    radii_viewpoints, nvs = sample_viewpoints()

    render_res = 512
    color = [0.5, 0.5, 0.5]
    background = torch.tensor([1.0, 1.0, 1.0]).cuda()

    # print(f"Reading the mesh...")
    mesh = Mesh(mesh_path)
    # print(f"Mesh faces shape:{mesh.faces.shape} and vertices shape:{mesh.vertices.shape}")

    # normalize mesh in unit sphere
    _ = MeshNormalizer(mesh)()

    # Coloring
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(torch.ones(1, len(mesh.vertices), 3).cuda() * torch.tensor(color).unsqueeze(0).unsqueeze(0).cuda(), mesh.faces)
    
    # Create the renderer
    # print(f"Creating the renderer...")
    render = Renderer(dim=(render_res, render_res))
    # print(f"Creating the renderer...done!")
      
    # Render the images 
    all_rendered_images = []
    all_rendered_images_faces_idx = []
    all_rendered_images_face_zs = []

    one_face_idx_shape = None
    for _, viewpoints in radii_viewpoints.items():
        n_views = viewpoints['n_views']
        elevs = torch.tensor(viewpoints['elevations'])
        azimuths = torch.tensor(viewpoints['azimuths'])
        radius = torch.tensor(viewpoints['radius'])

        rendered_images, elev, azim, masks, faces_idx, face_zs = render.render_views(
                                                                            elevs,
                                                                            azimuths,
                                                                            radius,
                                                                            mesh,
                                                                            num_views=n_views,
                                                                            show=False,
                                                                            center_azim=None,
                                                                            center_elev=None,
                                                                            std=None,
                                                                            return_views=True,
                                                                            return_mask=True,
                                                                            return_face_idx=True,
                                                                            lighting=True,
                                                                            background=background,
                                                                            seed=2023)
        faces_idx = torch.cat(faces_idx, dim=0)
        # if not one_face_idx_shape:
        #     one_face_idx_shape = faces_idx.shape
        face_zs = torch.cat(face_zs, dim=0)[:, :, :, -1].mean(dim=-1)

        all_rendered_images.append(rendered_images)
        all_rendered_images_faces_idx.append(faces_idx)
        all_rendered_images_face_zs.append(face_zs)

    all_rendered_images_raw = torch.cat(all_rendered_images, dim=0)
    all_rendered_images_faces_idx = torch.cat(all_rendered_images_faces_idx, dim=0)
    all_rendered_images_face_zs = torch.cat(all_rendered_images_face_zs, dim=0)
    
    # print(f'one face idx shape: {one_face_idx_shape}')
    # print(f'all face idx shape: {all_rendered_images_faces_idx.shape}')
    
    
    n_views = all_rendered_images_raw.shape[0]
    
    return all_rendered_images_faces_idx, n_views


# import math
# np.set_printoptions(threshold=math.inf)

dataset_dir = 'data/MPI-FAUST/training/sampled_scans/'
models = [f for f in os.listdir(dataset_dir) if osp.isfile(osp.join(dataset_dir, f)) and '_seg' not in f and 'obj' in f]

res = torch.empty((len(models), 120, 512, 512), dtype=torch.int16, device='cuda')
for i, m in tqdm(enumerate(models)):
    mesh_path = os.path.join(dataset_dir, m)
    mesh_name = m.split('.')[0]
    all_face_idx, n_views = get_face_idx(mesh_path)
    res[i] = all_face_idx

save_path = osp.join('./saved_renders', f'{args.max_elev_angle}.pt')
torch.save(res, save_path)
print(f'Saved all face indices to {save_path}')
print(f'Shape of saved renders: {res.shape}')
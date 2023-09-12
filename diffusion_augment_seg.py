#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import copy
import cv2
import torch
print(torch.__version__)
import scipy
import requests
import torch
import json
import trimesh
import einops
import torchvision
import numpy as np
import pandas as pd
import kaolin as kal
import os.path as osp
import kaolin.ops.mesh
import matplotlib.pyplot as plt

from mesh import Mesh
from render import Renderer
from tqdm.auto import tqdm
from collections import defaultdict
from IPython.display import display
from torchvision.ops import box_convert
from normalization import MeshNormalizer
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from pytorch_lightning import seed_everything


# In[2]:


##############################
# Include Grounded Segment Anything
# DINO / SAM 
##############################
# Please refer to the repo of Grounded Segment Anything here 
# https://github.com/IDEA-Research/Grounded-Segment-Anything
grounded_segment_anything_repo_path = "Grounded-Segment-Anything/"

sys.path.append(os.path.join(grounded_segment_anything_repo_path))
sys.path.append(os.path.join(grounded_segment_anything_repo_path, "GroundingDINO"))


# In[3]:


import GroundingDINO.groundingdino.datasets.transforms as T

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

from segment_anything import build_sam, build_sam_hq, SamPredictor 


# In[4]:


##############################
# Include ControlNet
##############################
controlnet_repo_path = 'ControlNet-v1-1-nightly/'
sys.path.append(controlnet_repo_path)


# In[5]:


import config

from share import *
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from annotator.zoe import ZoeDetector
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler



# Parse Arguments
import argparse
parser = argparse.ArgumentParser('Diffusion Augmented 3D Segmentation')
parser.add_argument('--max_elevation_angle', type=int, required=True)
parser.add_argument('--use_sam_hq', action='store_true')
parser.add_argument('--no_use_sam_hq', action='store_true')
parser.add_argument('--cache', type=int, required=True)
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--no_augment', action='store_true')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--control_type', type=str, required=True)

args = parser.parse_args()


# In[6]:


##############################
# Helper functions
##############################
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def sample_viewpoints(): # 120
    elevs = []
    azimuth = []
    
    min_radius = 1.5
    max_radius = 2.0
    
    # 5 different elevations (negatives and positive angles)
    if args.max_elevation_angle == 90:
        elevations = list((np.arange(start=0, stop=90, step=30)))
    elif args.max_elevation_angle == 75:
        elevations = list((np.arange(start=0, stop=75, step=25)))
    elif args.max_elevation_angle == 60:
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

# In[7]:


##############################
# Environment Variables 
##############################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[8]:


##############################
# Grounding DINO model
##############################

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)


# In[9]:


##############################
# Segment Anything Model
##############################
if args.use_sam_hq:
    sam = build_sam_hq(checkpoint='Grounded-Segment-Anything/sam_hq_vit_h.pth')
else:
    sam = build_sam(checkpoint='Grounded-Segment-Anything/sam_vit_h_4b8939.pth')

device = 'cuda:0'
sam.to(device=device)
sam_predictor = SamPredictor(sam)


# In[10]:


##############################
# Hyper-parameters
##############################
DINO_BOXES_THRESHOLD = 0.37
DINO_TEXT_THRESHOLD = 0.25


# In[11]:


PROMPT_COLOR =  {
'head': [1.0, 0.0, 0.0],
'torso': [0.0, 1.0, 0.0],
'arm': [1.0, 1.0, 0.0],
'leg': [0.0, 0.0, 1.0],
'forehead': [0.0, 1.0, 1.0],
'eye': [0.1, 0.5, 0.1],
'nose': [0.1, 0.1, 0.5],
'mouth': [0.5, 0.5, 0.1],
'chin': [0.2, 0.8, 0.7],
'ear': [0.6, 0.6, 0.9],
'neck': [0.9, 0.1, 0.9],
'belly_button': [0.9, 0.1, 0.1],
'knee': [0.9, 0.4, 0.1],
'foot': [0.5, 0.0, 0.9],
'elbow':[0.1, 0.5, 0.9],
'hand': [0.9, 0.2, 0.9],
'shoulder': [0.2, 0.2, 0.9],
'unknown' : [0.5, 0.5, 0.5],
'seat': [1.0, 0.0, 0.0],
'back': [0.0, 1.0, 0.0],
'wheel': [1.0, 0.0, 1.0],
'button':[1.0, 0., 0.],
'handle':[1., 1.0, 0.0],
'door': [0., 0., 1.],
'display': [0., 1.0, 0.0],
'blade':[1.0, 0., 0.],
'screw':[0., 0., 1.],
'thigh': [0., 1.0, 0.0],
'wrist': [1., 1.0, 0.0],
'limb':[1.0, 1.0, 0.0]
}

random_state = np.random.RandomState(2023)

def get_prompt_color(prompt):
    print(prompt)
    if prompt not in PROMPT_COLOR:
        PROMPT_COLOR[prompt] = [float(random_state.rand()), float(random_state.rand()), float(random_state.rand())]
    return PROMPT_COLOR[prompt] 


# In[12]:


##############################
# SAM-3D
##############################
class MeshTextSegmentor:
    def __init__(self,
                 mesh, 
                 n_views, 
                 prompts: list,
                 mesh_cls: str):
        
        assert len(prompts) > 0
        self.hightlighter = False
        self.mesh = mesh
        self.n_views = n_views
          
        self.faces_classes = np.zeros(
            (n_views, len(self.mesh.faces), len(prompts) + 1))  # +1 for others
            
        self.mesh_cls = mesh_cls
        self.prompts = list(prompts)
        self.prompt_str =  '. '.join(prompts) + ". " + mesh_cls + "."
        # self.prompt_str =  '. '.join(prompts) + "."
        
        self.prompt_to_id = {}
        self.id_to_prompt = {}
        self.output_res = None
        
        for i, p in enumerate(prompts):
            self.prompt_to_id[p] = i
            self.id_to_prompt[i] = p
        
        if self.hightlighter:
            self.face_classes_colors = [torch.tensor([204, 255, 0])]
        else:
            self.face_classes_colors = []
            for i in range(0, len(self.prompts)):  # +1 for being background or other
                col = get_prompt_color(self.prompts[i])
                self.face_classes_colors.append(
                    torch.tensor(col))
                print(self.prompts[i], col)

        self.prompt_to_id["background"] = len(prompts)
        self.id_to_prompt[len(prompts)] = "background"
        self.face_classes_colors.append(torch.tensor([180, 180, 180]))
        
        print(self.prompt_to_id)

    def segment_image(self, image_raw: np.ndarray):
        transformA = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
        
        transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        )
        
        image_source = np.asarray(image_raw * 255., dtype=np.uint8)
        image, _ = transform(transformA(image_source), None)
        
        # Run Dino model on the image
        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=self.prompt_str, 
            box_threshold=DINO_BOXES_THRESHOLD, 
            text_threshold=DINO_TEXT_THRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        
        # Run the segmentation model
        # set image
        sam_predictor.set_image(image_source)
        
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        
        if len(transformed_boxes) == 0:
            return [], [], [[]]
        
        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        t = [Image.fromarray(show_mask(el[0], annotated_frame)) for el in masks.cpu()]
        
        return masks, phrases, t

    def _compute_face_freq(self, f_inds):
        mentioned_f_ids, f_freqs = np.unique(
            f_inds.cpu().numpy(), return_counts=True)
        mentioned_f_freq_dict = np.zeros(len(self.mesh.faces))

        for k in range(len(mentioned_f_ids)):
            f_id = mentioned_f_ids[k]
            if f_id == -1:
                continue
            mentioned_f_freq_dict[f_id] = f_freqs[k]

        return mentioned_f_freq_dict

    def compute_segmentation_face_attributes(self, rendered_images, rendered_images_faces_idx):
        print(f"Segmenting using prompts: {self.prompts}")
        all_seg_masks = []
        all_ann_frames = []

        for i in tqdm(range(min(1000, len(rendered_images)))):
            # [N_prompts x 1 x 335 x 335], [N_prompts x 1 x 224 x 224]
            seg_masks, phrases, annotated_frame = self.segment_image(rendered_images[i])
            # print("Returned", len(x), [el == None for el in x])
            all_seg_masks.append(seg_masks)
            all_ann_frames.append(annotated_frame)

            # Now we need to add the segmentation information to the rendered faces
            rendered_img_face_ids = rendered_images_faces_idx[i]

            # Compute the freq of the faces (in terms of pixel count) rendered in this image
            rendered_face_freq_dict = self._compute_face_freq(
                rendered_img_face_ids)

            # Loop over the prompt's segmentation masks (each represents a segmentation class)
            # print(seg_masks.shape, phrases)
            n_masks = len(phrases)
            for j in range(n_masks):
                prompt = phrases[j]
                
                if prompt == self.mesh_cls:
                    continue
                
                if prompt not in  self.prompts:
                    continue
                
                if prompt not in self.prompt_to_id:
                    print(prompt)
                cls_id = self.prompt_to_id[prompt]

                mask = seg_masks[j].squeeze(0).cpu()
                
                self.output_res = rendered_images.shape[1]
                
                pixel_confs = mask.flatten()
                face_ids = rendered_img_face_ids.flatten()
                has_face = face_ids != -1
                filtered_face_id = face_ids[has_face]
                filtered_pixel_conf = pixel_confs[has_face]
                filetered_face_freq = rendered_face_freq_dict[filtered_face_id]
                tmp = filtered_pixel_conf *1.0 
                tmp = tmp.cpu().numpy()
                
                # print(self.faces_classes.shape, filtered_face_id.max(), cls_id)
                self.faces_classes[i, filtered_face_id, cls_id] += tmp
                
                

        confs = np.mean(self.faces_classes, axis=0)

        return confs, all_seg_masks, all_ann_frames

    def plot(self, ind, image, preds, prompts=None):
        if prompts is None:
            prompts = self.default_prompts

        _, ax = plt.subplots(1, len(prompts) + 1,
                             figsize=(3*(len(prompts) + 1), 4))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image.cpu().permute(1, 2, 0).numpy())
        ax[0].title.set_text(f'Rendered image: {ind}')
        [ax[i+1].imshow(preds[i][0]) for i in range(len(prompts))]
        [ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)]
        # plt.savefig(f'output/results{ind}.jpg', dpi=300)

    def color_mesh_and_save(self, sampled_mesh, cls_confs, output_filename="out.obj"):
        faces_not_assigned = np.where(np.sum(cls_confs, axis=-1) == 0)[0]

        face_cls = scipy.special.softmax(cls_confs, axis=-1)
        face_cls = np.argmax(face_cls, axis=-1)
        # face_cls = torch.softmax(torch.tensor(cls_confs), axis=-1).numpy()
        # face_cls = np.argmax(face_cls, axis=-1)

        face_cls[faces_not_assigned] = len(self.prompts)

        # print(f"We have {np.unique(face_cls)} as unique face classes")
        # for el in np.unique(face_cls):
            # print(self.id_to_prompt[el])

        # Now color the facees according to the predicted class
        face_colors = torch.zeros(
            (len(sampled_mesh.faces),  4), dtype=torch.uint8)
        
        # Return the class per vertex
        vertex_class = [''] * len(sampled_mesh.vertices)
        prompts = self.prompts + ['background']
        
        for i in range(len(sampled_mesh.faces)):
            face_colors[i][0:3] = self.face_classes_colors[face_cls[i]] * 255.
            face_colors[i][-1] = 255
            
            for j in range(3):
                v_i = sampled_mesh.faces[i][j]
                vertex_class[v_i] = prompts[face_cls[i]]
                
        # Create trimesh
        scene = trimesh.Scene()
        output_mesh = trimesh.Trimesh(vertices=sampled_mesh.vertices.cpu(
        ).numpy(), faces=sampled_mesh.faces.cpu().numpy())
        output_mesh.visual.face_colors = face_colors.cpu().numpy()
        scene.add_geometry(output_mesh, node_name='output')
        out = scene.export(output_filename)
        
        with open(f"{output_filename.replace('obj', 'json')}", 'w') as fout:
            json.dump(vertex_class, fout)


# In[13]:
# model_name = 'control_v11f1p_sd15_depth'
model_name = 'control_v11p_sd15_canny'

class ControlNetDepth:
    def __init__(self):
        model = create_model(f'ControlNet-v1-1-nightly/models/{model_name}.yaml').cpu()
        if not osp.isfile('ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt'):
            import requests
            response = requests.get('https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt')
            if response.status_code == 200:
                with open('ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt', 'wb') as file:
                    file.write(response.content)
                    print('Downloaded weights to: ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt')
            else:
                print('Failed to download ControlNet weights')
                
        if not osp.isfile(f'ControlNet-v1-1-nightly/models/{model_name}.pth'):
            import requests
            response = requests.get('https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth')
            if response.status_code == 200:
                with open(f'ControlNet-v1-1-nightly/models/{model_name}.pth', 'wb') as file:
                    file.write(response.content)
                    print(f'Downloaded weights to: ControlNet-v1-1-nightly/models/{model_name}.pth')
            else:
                print('Failed to download ControlNet weights')
                    
        model.load_state_dict(load_state_dict('ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        model.load_state_dict(load_state_dict(f'ControlNet-v1-1-nightly/models/{model_name}.pth', location='cuda'), strict=False)
        
        self.preprocessor = None
        self.model = model.cuda()
        self.ddim_sampler = DDIMSampler(model)
        
        # Configs
        self.prompt = "high quality image, human, simple background, angled view"
        self.num_samples = 1
        self.seed = 2023
        self.det = 'Canny'  # Possible values ["Depth_Zoe", "Depth_Midas", "None"]
        self.image_resolution = 512
        self.strength = 1.0
        self.guess_mode = False
        self.detect_resolution = 512
        self.ddim_steps = 20 # finetune this
        self.scale = 9.0
        self.eta = 1
        self.a_prompt = 'best quality, best anatomy'
        self.n_prompt = 'lowres, bad anatomy, bad hands, bad feet, cropped, worst quality, shadows, clothes, fur, bright lights, distortions, transparent'
        
        self.low_threshold = 100
        self.high_threshold = 200
        
    def process(self, input_image):
        if self.det == 'Depth_Midas':
            if not isinstance(self.preprocessor, MidasDetector):
                self.preprocessor = MidasDetector()
        if self.det == 'Depth_Zoe':
            if not isinstance(self.preprocessor, ZoeDetector):
                self.preprocessor = ZoeDetector()
                
        if self.det == 'Canny':
            if not isinstance(self.preprocessor, CannyDetector):
                self.preprocessor = CannyDetector()

        with torch.no_grad():
            input_image = HWC3(input_image)

            if self.det == 'None':
                detected_map = input_image.copy()
            else:
                # detected_map = self.preprocessor(resize_image(input_image, self.detect_resolution))
                detected_map = self.preprocessor(resize_image(input_image, self.detect_resolution), self.low_threshold, self.high_threshold)
                detected_map = HWC3(detected_map)

            img = resize_image(input_image, self.image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(self.num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            seed_everything(self.seed)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([self.prompt + ', ' + self.a_prompt] * self.num_samples)]}
            un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
            shape = (4, H // 8, W // 8)

            self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

            samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples,
                                                         shape, cond, verbose=False, eta=self.eta,
                                                         unconditional_guidance_scale=self.scale,
                                                         unconditional_conditioning=un_cond)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(self.num_samples)]

        return [detected_map] + results


# In[14]:


controlNetModel = ControlNetDepth()


# In[15]:


output_dir = f'cache{args.cache}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(osp.join(output_dir, 'rendered_images'), exist_ok=True)
os.makedirs(osp.join(output_dir, 'augmented_images'), exist_ok=True)


# In[16]:


def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out

def get_depth_maps (face_ids, face_depths):
    # print("Face ids, face depth")
    # print(face_ids.shape, face_depths.shape)
    # face_ids N_views x H x W
    # face_depths N_views x N_faces
    ret = torch.zeros_like(face_ids)
    
    n_views, h, w = face_ids.shape
    face_ids_flattened = face_ids.view(n_views, h*w)
    mask = face_ids != -1
    mask_neg = face_ids_flattened == -1 # N_views x (H x W)
    mask_pos = face_ids_flattened != -1 # N_views x (H x W)
    face_ids_flattened[mask_neg] = 0 # Avoid cuda run time error and we will mask it later
    depths = vector_gather(face_depths.unsqueeze(-1), face_ids_flattened)
    depths = depths.squeeze(-1)
    depths *= mask_pos
    
    # normalize per view, the current range is [-inf, 0]
    z_min = depths.min(dim=1).values.unsqueeze(-1)
    depths = depths / ((z_min) + 0.00000001)
    
    depths = 1.0 - depths.view(n_views, h, w)
    depths *= mask
    
    return depths
    

def segmentMesh(mesh_path, prompts, mesh_cls, saveDir, meshName, augment=True, n_augm_views=1):
    # if args.image_path:
    #     mesh = Mesh(mesh_path)
    #     _ = MeshNormalizer(mesh)()
    #     images = []
        
    #     for i in range(120):
    #         images.append(torch.tensor(np.array(Image.open(os.path.join(args.image_path, meshName, f'{i}.png')))).unsqueeze(0))
    #     images = torch.cat(images, dim=0).cuda() / 255.
        
        
        
        
    
    prompts = list(set(prompts))
    
    prompts_colos = []
    radii_viewpoints, nvs = sample_viewpoints()

    render_res = 512
    color = [0.5, 0.5, 0.5]
    background = torch.tensor([1.0, 1.0, 1.0]).cuda()

    print(f"Reading the mesh...")
    mesh = Mesh(mesh_path)
    print(f"Mesh faces shape:{mesh.faces.shape} and vertices shape:{mesh.vertices.shape}")

    # normalize mesh in unit sphere
    _ = MeshNormalizer(mesh)()

    # Coloring
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(torch.ones(1, len(mesh.vertices), 3).cuda() * torch.tensor(color).unsqueeze(0).unsqueeze(0).cuda(), mesh.faces)
    
    # Create the renderer
    print(f"Creating the renderer...")
    render = Renderer(dim=(render_res, render_res))
    print(f"Creating the renderer...done!")
      
    # Render the images 
    all_rendered_images = []
    all_rendered_images_faces_idx = []
    all_rendered_images_face_zs = []

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
        face_zs = torch.cat(face_zs, dim=0)[:, :, :, -1].mean(dim=-1)

        all_rendered_images.append(rendered_images)
        all_rendered_images_faces_idx.append(faces_idx)
        all_rendered_images_face_zs.append(face_zs)

    all_rendered_images_raw = torch.cat(all_rendered_images, dim=0)
    all_rendered_images_faces_idx = torch.cat(all_rendered_images_faces_idx, dim=0)
    all_rendered_images_face_zs = torch.cat(all_rendered_images_face_zs, dim=0)
    
    
    n_views = all_rendered_images_raw.shape[0]
    
    all_rendered_images_face_zs += 1.0
    depth_maps = get_depth_maps(all_rendered_images_faces_idx, all_rendered_images_face_zs)
    
    augmented_images = []
    final_images = []
    for i in range(n_views):
        if not osp.isdir(osp.join(output_dir, 'augmented_images', meshName)):
            os.makedirs(osp.join(output_dir, 'augmented_images', meshName), exist_ok=True)
        if not osp.isdir(osp.join(output_dir, 'rendered_images', meshName)):
            os.makedirs(osp.join(output_dir, 'rendered_images', meshName), exist_ok=True)
        if not osp.isdir(osp.join(output_dir, 'depth_images', meshName)):
            os.makedirs(osp.join(output_dir, 'depth_images', meshName), exist_ok=True)
        
        if augment:
            if osp.isfile(osp.join(output_dir, 'augmented_images', meshName, f"{i}.png")):
                pass
            else:
                if args.control_type == 'depth':
                    dm = controlNetModel.process(np.array(depth_maps[i].cpu().numpy() * 255., dtype=np.uint8))
                elif args.control_type == 'canny':
                    dm = controlNetModel.process(np.array(all_rendered_images_raw[i].permute(1, 2, 0).cpu().numpy() * 255., dtype=np.uint8))
                resultImage = Image.fromarray(dm[1])
                resultImage.save(osp.join(output_dir, 'augmented_images', meshName, f"{i}.png"))
                # augmented_images.append((torch.tensor(np.array(resultImage) / 255.).unsqueeze(0)))
        
            augmented_images.append(torch.tensor(np.array(Image.open(osp.join(output_dir, 'augmented_images', meshName, f"{i}.png")))).unsqueeze(0))
        else:
            if not osp.isfile(osp.join(output_dir, 'rendered_images', meshName, f"{i}.png")):
                resultImage = Image.fromarray(np.array(all_rendered_images_raw[i].permute(1, 2, 0).cpu().numpy() *255., dtype=np.uint8))
                resultImage.save(osp.join(output_dir, 'rendered_images', meshName, f"{i}.png"))
            #final_images.append(torch.tensor(np.array(Image.open(osp.join(output_dir, 'rendered_images', meshName, f"{i}.png")))).unsqueeze(0))
        if not osp.isfile(osp.join(output_dir, 'depth_images', meshName, f"{i}.png")):
            resultImage = Image.fromarray(np.array(depth_maps[i].cpu().numpy() *255., dtype=np.uint8))
            resultImage.save(osp.join(output_dir, 'depth_images', meshName, f"{i}.png"))
    
    if augment:
        augmented_images = torch.cat(augmented_images, dim=0).cuda() / 255.
        augmented_images = augmented_images * (all_rendered_images_faces_idx.unsqueeze(-1) != -1)
        final_images = augmented_images
    else:
        final_images = all_rendered_images_raw.permute(0,2,3,1)

    # now Segment
    segmentor = MeshTextSegmentor(
            mesh, n_views=len(final_images), prompts=prompts, mesh_cls=mesh_cls)
    
    cnfs, all_seg_masks, all_ann_frames = segmentor.compute_segmentation_face_attributes(rendered_images=final_images.cpu().numpy(), rendered_images_faces_idx=all_rendered_images_faces_idx.cpu())
        
    
    os.makedirs(os.path.join(saveDir, meshName), exist_ok=True)
    output_filename = os.path.join(saveDir, meshName, 'output.obj')
    prompts_output_filename = os.path.join(saveDir, meshName, 'prompts.txt')
    
    with open(prompts_output_filename, 'w') as fout:
        json.dump(prompts, fout)

    # Export the segementation information
    segmentor.color_mesh_and_save(
        mesh, cnfs, output_filename)


# In[17]:


# Now segment FAUST dataset (SATR's version?)
dataset_dir = 'data/MPI-FAUST/training/sampled_scans/'
models = [f for f in os.listdir(dataset_dir) if osp.isfile(osp.join(dataset_dir, f)) and '_seg' not in f and 'obj' in f]
print(len(models))


saveDir = f'cache{args.cache}/sam3D_dino_{args.experiment_name}'


# Now assume coarse grained
from time import time
for i, modelName in tqdm(enumerate(models)):
    # try:
        mesh = modelName
        meshName = mesh.split('.')[0]

        # Segmenet mesh A
        # saveDir = f'cache2/sam3D_dino_a1'

        mesh_path =os.path.join(dataset_dir, mesh)
        prompts = ["arm", "head", "torso", "leg"]
        mesh_cls = 'person'

        if not os.path.exists(os.path.join(saveDir, meshName, 'output.obj')):
            print("Segmenting", meshName)
            start = time()
            if args.augment:
                segmentMesh(mesh_path, prompts, mesh_cls, saveDir, meshName, augment=True)
            else:
                segmentMesh(mesh_path, prompts, mesh_cls, saveDir, meshName, augment=False)
            end = time()
            minutes, seconds = divmod(end-start,60)
            print(f'Segmentation for {meshName} took {end-start} seconds, or {minutes} minutes and {seconds} seconds')

    # except:
        # print(f"Error for model:{modelName}")


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


# In[ ]:


# Now evaluate the clipseg performance 

# Read the predicted cls for each mesh
# Read the ground-truth 
dataDir = osp.join('data/MPI-FAUST/training/sampled_scans/')
partsIous = defaultdict(list)

for i, m in tqdm(enumerate(models)):
    mesh_path = os.path.join(dataDir, m)
    mesh_name = m.split('.')[0]
    output_dir = os.path.join(saveDir, mesh_name)
    
    with open(os.path.join(output_dir, f"output.json")) as fin:
        pred = np.array(json.load(fin))
    
    with open(os.path.join(f"final_coarse.json")) as fin:
        gt = np.array(json.load(fin)[f'{mesh_name}'])

    assert 'unknown' not in gt
    gtLabels = sorted(set(gt))

    for l in gtLabels:
        partsIous[l].append(calculate_shape_IoU(pred, gt, l))
    assert(len(pred) == len(gt))
    
    


# In[ ]:


# calculate mIoU
mIous = []
for k, v in partsIous.items():
    mIous.append(np.mean(v))

print(np.mean(mIous))


# In[ ]:


def get_seg_label(image_face_ids, gt_verts_labels, mesh):
    # Get the gt face labels by parsing the vertices labels
    mesh_face_labels = np.zeros(len(mesh.faces))
    for i_f, face_verts in enumerate(mesh.faces):
        mesh_face_labels[i_f] = gt_verts_labels[face_verts[0]]
    
    # Now you know the gt labels for each face in the mesh, you can now generate a matrix of the same size of face_ids matrix but labeled
    image_face_labels = [[] * image_face_ids.shape[0]]
    for i_row in range(image_face_ids.shape[0]):
        for i_col in range(image_face_ids.shape[1]):
            face_label = 'unknown' if image_face_ids[i_row][i_col] == -1 else mesh_face_labels[image_face_ids[i_row][i_col]]
            image_face_labels[i_row].append(face_label)

    return image_face_labels


# In[ ]:





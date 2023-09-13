import os
from os import path as osp
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict

from huggingface_hub import hf_hub_download
from pytorch_lightning import seed_everything

grounded_segment_anything_repo_path = "Grounded-Segment-Anything/"

sys.path.append(os.path.join(grounded_segment_anything_repo_path))
sys.path.append(os.path.join(grounded_segment_anything_repo_path, "GroundingDINO"))

import GroundingDINO.groundingdino.datasets.transforms as T

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

from segment_anything import build_sam, build_sam_hq, SamPredictor 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--caches', type=str, required=True)

'13o,13a,14o,14a'

args = parser.parse_args()
target_list = [f'cache{s[:len(s)-1]}/augmented_images' if s[-1] == 'a' else \
               f'cache{s[:len(s)-1]}/rendered_images' \
               for s in args.caches.split(',')]

print(target_list)




os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
sam = build_sam_hq(checkpoint='Grounded-Segment-Anything/sam_hq_vit_h.pth')
device = 'cuda:0'
sam.to(device=device)
sam_predictor = SamPredictor(sam)
DINO_BOXES_THRESHOLD = 0.37
DINO_TEXT_THRESHOLD = 0.25
prompts = ["arm", "head", "torso", "leg"]
mesh_cls = 'person'
prompt_str =  '. '.join(prompts) + ". "

def predict_bbox(image_path: str):
        transformA = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
        
        transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        )
        
        image_source = np.asarray(Image.open(image_path), dtype=np.uint8)
        
        # image_source = np.asarray(image_raw * 255., dtype=np.uint8)
        image, _ = transform(transformA(image_source), None)
        
        # Run Dino model on the image
        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=prompt_str, 
            box_threshold=DINO_BOXES_THRESHOLD, 
            text_threshold=DINO_TEXT_THRESHOLD
        )

        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        
        # Run the segmentation model
        # set image
        # sam_predictor.set_image(image_source)
        
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        
        if len(transformed_boxes) == 0:
            return [], [], [[]]
        
        # masks, _, _ = sam_predictor.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes,
        #     multimask_output = False,
        # )
        
        # t = [Image.fromarray(show_mask(el[0], annotated_frame)) for el in masks.cpu()]
        
        # return masks, phrases, t
        return phrases, boxes_xyxy
    
# print(predict_bbox('./cache11/rendered_images/tr_scan_000/0.png'))

dataset_dir = 'data/MPI-FAUST/training/sampled_scans/'
cls = 'person'
labels = sorted(['head','arm','torso','leg'])
models = [f for f in os.listdir(dataset_dir) if osp.isfile(osp.join(dataset_dir, f)) and '_seg' not in f and 'obj' in f]

for target_path in target_list:
    all_phrases_to_boxes = []
    for i,m in tqdm(enumerate(models)):
        mesh_path = os.path.join(dataset_dir, m)
        mesh_name = m.split('.')[0]
        images_dir = os.path.join(target_path, mesh_name)
        
        phrases_to_boxes = []
        for image_path in os.listdir(images_dir):
            p2b = dict()
            phrases, boxes = predict_bbox(image_path)
            for p,b in zip(phrases, boxes):
                if p in p2b:
                    xMinNew, yMinNew, xMaxNew, yMaxNew = boxes
                    xMinOld, yMinOld, xMaxOld, yMaxOld = p2b[p]
                    p2b[p] = torch.tensor([min(xMinNew,xMinOld), min(yMinNew,yMinOld),
                                           max(xMaxNew,xMaxOld), max(yMaxNew, yMaxOld)])
                else:
                    p2b[p] = b
            for l in labels:
                if l not in p2b.keys():
                    p2b[l] = None
            phrases_to_boxes.append(p2b)
        
        all_phrases_to_boxes.append(phrases_to_boxes)
    torch.save(all_phrases_to_boxes, osp.join('./saved_bboxes', ''.join([x for x in target_path if x != '/'])))
# This is a version of READMem+baseline (MiVOS) adapted for the vots2023 challenge.
# Includes the integration for the challenge

# by Stéphane Vujasinovic: begin (06.06.2023) - end ()

# -----------------
# - Import packages
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from MiVOS.model.propagation.prop_net import PropagationNetwork
from inference_core_MiVOS_HoliMem import InferenceCore

from torchvision import transforms

import os
import sys

sys.path.append('/home/ste03528/WORK_Station/VOTS') # ABSOLUTE PATH TO THE FOLDER CONTAININT THE INTEGRATION PACKAAGE FROM VOT: git clone https://github.com/votchallenge/integration
import integration.python.vot as vot_py

# ------------------------------------------
# - LOADING CHECKPOINTS AND CONFIGURATIONS -
print(f'Current directory: {os.getcwd()}')
# ABSOLUTE PATH TO THE MIVOS PROPAGATION MODEL WEIGHTS
pseudo_arg_model = '/home/ste03528/WORK_Station/VOTS/workspace_test/Tracker_READMem_MiVOS/MiVOS/saves/propagation_model.pth'
# ABSOLUTE PATH TO THE CONFIGURATION FILE
pseudo_arg_mem_configuration = '/home/ste03528/WORK_Station/VOTS/workspace_test/Tracker_READMem_MiVOS/config_tracker.yaml'

# MiVOS intrinsic configs.
pseudo_no_top = False
pseuoo_record_det = False
pseudo_flip = False

torch.autograd.set_grad_enabled(False)

# Load the MiVOS model
prop_saved = torch.load(pseudo_arg_model)
top_k = None if pseudo_no_top else 50
prop_model = PropagationNetwork(top_k=top_k).cuda().eval()
prop_model.load_state_dict(prop_saved)
record_deter_flag = False

# Other configs
PSEUDO_MIN_SIZE = 480
im_mean = (124, 116, 104)

# ---------------------------------
# - FUNCTIONS DECLARATION AND ETC -
# Transformation declaration
im_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
im_transform= transforms.Compose([
    transforms.ToTensor(),
    im_normalization
])

im_transform_and_resize = transforms.Compose([
    transforms.ToTensor(),
    im_normalization,
    transforms.Resize(PSEUDO_MIN_SIZE, interpolation=Image.BICUBIC),   # Resizing
])

# Only functions
def handle_img_for_tracker(img_path):
    img_np = Image.open(img_path).convert('RGB')
    mask_img = np.zeros([len(objects), 1, *np.array(img_np).shape[:2]])
    min_hw = min(img_np.size)

    if PSEUDO_MIN_SIZE >= min_hw:
        return im_transform(img_np), mask_img, img_np.size[::-1]

    else:
        return im_transform_and_resize(img_np), mask_img, img_np.size[::-1]

def resize_mask(mask):
    h, w = mask.shape[-2:]
    min_hw = min(h, w)
    mask_Tensor = F.interpolate(mask, (1, int(h/min_hw*PSEUDO_MIN_SIZE), int(w/min_hw*PSEUDO_MIN_SIZE)),
                mode='nearest')

    return mask_Tensor

def bring_back_to_orignal_size_mask(mask, original_size):
    mask = mask.type(torch.FloatTensor)
    mask = F.interpolate(mask.unsqueeze(dim=0), original_size, mode='bilinear', align_corners=False)[:, 0]

    return mask

def mask_from_tensor_2_numpy(mask, original_size):
    msk = torch.argmax(mask, dim=0)
    if PSEUDO_MIN_SIZE <= min(msk.shape[-2:]):
        msk = bring_back_to_orignal_size_mask(msk, original_size)

    return (msk.detach().cpu().numpy()).astype(np.uint8)[0]


# -------------------
# - VOT INTEGRATION -
handle = vot_py.VOT("mask", multiobject=True)
objects = handle.objects()
imagefile = handle.frame()
image, mask_img, original_size = handle_img_for_tracker(imagefile)

# Adapt the mask for the Tracker
for idx, msk in enumerate(objects):
    H,W = msk.shape
    mask_img[idx, 0, 0:H,0:W] = msk

msk = torch.Tensor(mask_img).cuda()
if PSEUDO_MIN_SIZE <= min(msk.shape[-2:]):
    msk = resize_mask(msk.unsqueeze(0))[0]

# ---------------------------
# - INITIALIZE THE TRACKING -
VID_LENGTH = 10000000   # TODO: don't need this, discard it
kdx = 0                 # frame index
NBR_OF_OBJECTS = 1
trackers = [InferenceCore(prop_model, NBR_OF_OBJECTS, pseudo_arg_mem_configuration, False, False, False) for _ in objects]  # Create the trackers
out_masks = [tracker.set_annotated_frame(kdx, VID_LENGTH, image, mask) for tracker, mask in zip(trackers, msk)]             # Initialize the trackers with the first image

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image, _, original_size = handle_img_for_tracker(imagefile)
    kdx += 1

    # Tracking
    handle.report([mask_from_tensor_2_numpy(tracker.step(kdx, image), original_size) for tracker in trackers])




#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
from model import log

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

#get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "./logs/coco20210709T0922/mask_rcnn_coco_0159.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "tests")
#IMAGE_DIR = '/datasets/coco/val2017'
IMAGE_DIR = './tests/'


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inference_config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)



import cv2
# COCO Class names
#For human pose task We just use "BG" and "person"
class_names = ['BG', 'person']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
#image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
for file_name in file_names:
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = cv2.imread(os.path.join(IMAGE_DIR, file_name))
    #BGR->RGB
    image = image[:,:,::-1]

    # Run detection
    results = model.detect_keypoint([image], verbose=1)
    r = results[0] # for one image
    if len(r['rois'])==0:
        continue

    log("rois",r['rois'])
    log("keypoints",r['keypoints'])
    log("class_ids",r['class_ids'])
    log("keypoints",r['keypoints'])
    log("masks",r['masks'])
    log("scores",r['scores'])

    #visualize.display_keypoints(image,r['rois'],r['keypoints'],r['class_ids'],class_names,
    #        skeleton = inference_config.LIMBS,
    #        save_dir= 'tmp', image_name = file_name)
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                            class_names, r['scores'], save_dir = 'tmp_mask', image_name = file_name)
    visualize.display_keypoints_and_instance(image,r['rois'], r['keypoints'],
            r['masks'], r['class_ids'], class_names,
            skeleton = inference_config.LIMBS, scores=r['scores'],
            save_dir= 'tmp_mask', image_name = file_name)


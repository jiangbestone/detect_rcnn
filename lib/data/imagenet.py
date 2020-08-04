from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------



import os, sys
import numpy as np
import scipy.sparse
import scipy.io as sio
import pdb
import pickle




def get_init():
    # Hyperparameters
    hyp = {'init_lr': 0.001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 5e-4,  # optimizer weight decay
           'giou': 0.05,  # giou loss gain
           'cls': 0.58,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.20,  # iou training threshold
           'anchor_t': 4.0,  # anchor-multiple threshold
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 0.0,  # image rotation (+/- deg)
           'translate': 0.0,  # image translation (+/- fraction)
           'scale': 0.5,  # image scale (+/- gain)
           'shear': 0.0}  # image shear (+/- deg)
    return hyp





import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# from transformations import euler_from_matrix
from random import random
import time
# import
from semantic_boxes import get_unique_colors, get_blobs_and_boxes, build_image_graph, get_sep_category_imgs, remove_noise
# get_unique_colors , get_sep_category_imgs




def category_wise_boxes(real_img, img_seg):
    colors = get_unique_colors(img_seg)
    sep_seg_imgs = get_sep_category_imgs(img_seg)

    sep_denoised = []
    for sep_img in sep_seg_imgs:
        if(sep_img is None):
            sep_denoised.append(None)
            continue
        else:
            denoised = remove_noise(sep_img)
            sep_denoised.append(denoised)
    categ_wise_boxes = get_blobs_and_boxes(sep_denoised,real_img)
    return categ_wise_boxes

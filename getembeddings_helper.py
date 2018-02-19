import cv2
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from getembeddings import getembeddings
import sys
from getguassianprojections import getguassianprojections
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, './caffetensorflow')
# sys.path.append("./models/research/slim")

# from datasets import dataset_utils
# from preprocessing import vgg_preprocessing


# img_i =  cv2.imread('i1.png')
# image = img_i
#
# i = []
# i.append(img_i)
# # i.append(img_i)
#
# image = np.asarray(i)

image1 =  cv2.imread('i1.png')
image2 =  cv2.imread('i2.png')

# ----------------------------------------------------------------------------------------------
# image_size = vgg.vgg_16.default_image_size
# processed_image = vgg_preprocessing.preprocess_image(i, image_size,image_size,is_training=False)
model_name = 'inception'
layer_name = 'Mixed_5b'

image = image1
a = getembeddings(image, model_name, layer_name)
# print(a.shape)
n_components = 100
a_afterprojections = getguassianprojections(a, n_components)
a1 = a_afterprojections


image = image2
a = getembeddings(image, model_name, layer_name)
# print(a.shape)
n_components = 100
a_afterprojections = getguassianprojections(a, n_components)
a2 = a_afterprojections

d_cos = cosine_distances(a1, a2)


# ----------------------------------------------------------------------------------------------

# model_name = 'places365vgg'
# layer_name = 'fc6'
#
# b = getembeddings(image, model_name, layer_name)
# print(b.shape)
#
# n_components = 100
# # n_components = 'auto'
# b_afterprojections = getguassianprojections(b, n_components)

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



inception_model_path = "./inception_v3.ckpt"
vgg_model_path = "./vgg_16.ckpt"
resnet_model_path = "./resnet_v1_101.ckpt"
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
model_path = inception_model_path


image = image1
a = getembeddings(image, model_name, layer_name, model_path)
# print(a.shape)
a1 = a.reshape(1,-1 )
# n_components = 100
# a_afterprojections = getguassianprojections(a, n_components)
# a1f = a_afterprojections


image = image2
a = getembeddings(image, model_name, layer_name, model_path)
# print(a.shape)
a2 = a.reshape(1,-1 )

# n_components = 100
# a_afterprojections = getguassianprojections(a, n_components)
# a2f = a_afterprojections

d_cos = cosine_distances(a1, a2)
print(d_cos)
li.append(d_cos[0][0])

# ----------------------------------------------------------------------------------------------

model_name = 'places365vgg'
layer_name = 'fc6'
model_path = './mynet_backup.npy'

b = getembeddings(image, model_name, layer_name, model_path)
print(b.shape)
#
# n_components = 100
# # n_components = 'auto'
# b_afterprojections = getguassianprojections(b, n_components)




# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# model_name = 'inception'
# layer_name = 'Mixed_5b'
# model_path = inception_model_path
#
# model_name = 'vgg'
# layer_name = 'vgg_16/pool4'
# model_path = vgg_model_path
#
#
# model_name = 'resnet'
# layer_name = 'resnet_v1_101/block3/unit_2/bottleneck_v1/conv1'
# model_path = resnet_model_path

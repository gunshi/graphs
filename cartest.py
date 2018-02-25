import cv2
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from getembeddings import getembeddings
import sys
from embeddings import Embeddings
from os import listdir
from os.path import isfile, join

from sklearn.metrics.pairwise import cosine_distances


# model_name = 'places365vgg'
# layer_name = 'prob'
# model_path = './mynet.npy'


model_name = 'places365resnetft'
layer_name = 'bn5a_branch2c'
model_path = './resnet365ft.npy'


# model_name = 'places365resnet'
# layer_name = 'prob'
# model_path = './resnet365.npy'

path1 = "/run/user/1000/gvfs/sftp:host=10.2.36.75,user=anjan/tmp/anjan/2014-05-06-12-54-54/stereo/centre_corrected/"
path2 = "/run/user/1000/gvfs/sftp:host=10.2.36.75,user=anjan/tmp/anjan/2014-05-06-12-54-54/mono_rear_corrected/"
clf = Embeddings(model_name, layer_name, model_path)


imagenames1 = [f for f in listdir(path1)]
imagenames2 = [f for f in listdir(path2)]
imagenames1.sort()
imagenames2.sort()
images1 = [join(path1, i) for i in imagenames1]
images2 = [join(path2, i) for i in imagenames2]

temp1 = [[imagenames1[i], imagenames2[i]] for i in range(25)]

temp = [[int(imagenames1[i][:-4]) - int(imagenames1[i-1][:-4]), int(imagenames2[i][:-4]) - int(imagenames2[i-1][:-4])] for i in range(1, 50)]

timestamps1 = np.asarray([int(i[:-4]) for i in imagenames1])
timestamps2 = np.asarray([int(i[:-4]) for i in imagenames2])

diff = int(imagenames2[588][:-4]) - int(imagenames1[29][:-4])
ind = np.searchsorted(timestamps2, timestamps1 + diff)

# 29 in 1, 588 in 2
start = 29
for i in range(start, len(imagenames1)):
    i1 =  cv2.imread(images1[i])
    i2 =  cv2.imread(images2[ind[i]])
    a1 = clf.run(i1)
    a2 = clf.run(i2)
    d_cos = cosine_similarity(a1.reshape(1,-1 ), a2.reshape(1,-1 ))
    print(d_cos)

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def cosine_patch_distance(feature1, width1, height1, feature2, width2, height2):

    if(len(feature1.shape) ==1):
        feature1 = np.expand_dims(feature1, axis=0)
    if(len(feature2.shape) ==1):
        feature2 = np.expand_dims(feature2, axis=0)
    d = cosine_distances(feature1, feature2)
    d_cosine = d[0][0]

    # shape similarity measure
    d_shape =  np.exp(0.5 * (np.abs(width1 - width2)/np.maximum(width1, width2) + np.abs(height1 - height2)/np.maximum(height1, height2)))

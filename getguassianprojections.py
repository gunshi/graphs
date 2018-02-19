import numpy as np
from sklearn import random_projection


def getguassianprojections(features, n_components='auto'):
    features_reshaped = features.reshape(features.shape[0] , -1)
    X = features_reshaped
    transformer = random_projection.GaussianRandomProjection(n_components=n_components)
    X_new = transformer.fit_transform(X)
    print(X_new.shape)
    return X_new

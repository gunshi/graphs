import cv2
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, './caffetensorflow')
# sys.path.append("./models/research/slim")
#
# from datasets import dataset_utils
# from preprocessing import vgg_preprocessing


def preprocess(image, model_name, inplace= False):

    if(inplace):
        image1 = image
    else:
        image1 = np.copy(image)

    if(model_name is 'inception'):
        print(model_name)
        image1 = (2.0) * ((image1/255)-0.5)
        return image1

    elif(model_name is 'resnet'):
        print(model_name)
        return image1
        image1 = (2.0) * ((image1/255)-0.5)

    elif(model_name is 'vgg'):
        print(model_name)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        mean = [_R_MEAN, _G_MEAN, _B_MEAN]

        image1[:, :, 0] = image1[:, :, 0] - mean[0]
        image1[:, :, 1] = image1[:, :, 1] - mean[1]
        image1[:, :, 2] = image1[:, :, 2] - mean[2]
        return image1

    elif(model_name is 'places365vgg'):
        print(model_name)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        mean = [_B_MEAN, _G_MEAN, _R_MEAN]

        image1 = image1[:,:,::-1]
        image1[:, :, 0] = image1[:, :, 0] - mean[0]
        image1[:, :, 1] = image1[:, :, 1] - mean[1]
        image1[:, :, 2] = image1[:, :, 2] - mean[2]
        return image1


    else:
        return image1

def getembeddings(image, model_name, layer_name, model_path):
    # --------------------------------------------------------------------------------------------------------------------------------------
    tf.reset_default_graph()

    # --------------------------------------------------------------------------------------------------------------------------------------
    model_dicts = {'resnet' : {
                                'image_size' : 224 ,
                                'model' : "nets.resnet_v1.resnet_v1_101" ,
                                'scope' : "nets.resnet_v1.resnet_arg_scope()",
                                "num_classes" : 1000
                            },
                'inception' : {
                                'image_size' : 299 ,
                                'model' : "nets.inception.inception_v3",
                                'scope' : "nets.inception.inception_v3_arg_scope()" ,
                                "num_classes" : 1001
                            },

                'vgg' : {
                                'image_size' : 224 ,
                                'model' : "nets.vgg.vgg_16",
                                'scope' : "nets.vgg.vgg_arg_scope()" ,
                                "num_classes" : 1000
                            }

                }





    # --------------------------------------------------------------------------------------------------------------------------------------
    flag = 0
    if(model_name is 'places365vgg'):
        # from  mynet import VGGPlaces365 as MyNet
        from  VGGPlaces365 import VGGPlaces365 as MyNet


        image_size = 224
        images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        net = MyNet({'data':images})
        layer_name = 'fc6'
        layer_features = net.layers[layer_name]
    else:
        flag = 1
        if (type(model_name) is str and model_name in model_dicts):
            model_entry = model_dicts[model_name]
        else:
            model_entry = model_name

        net = eval(model_entry['model'])
        image_size = model_entry['image_size']

        images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])

        with slim.arg_scope(eval(model_entry['scope'])):
            last_layer_logits, end_points = net(images, num_classes=model_entry['num_classes'])
            layer_features = end_points[layer_name]



    # --------------------------------------------------------------------------------------------------------------------------------------
    image=image.astype('float32')

    if(len(image.shape) ==3 ):
        reshapedimage = preprocess(cv2.resize(image,(image_size, image_size), interpolation = cv2.INTER_CUBIC), model_name)
        reshapedimageinput = np.expand_dims(reshapedimage, axis=0)

    else:
        reshapedimageinput = []
        for i in image:
            reshapedimageinput.append(preprocess( cv2.resize(i,(image_size, image_size), interpolation = cv2.INTER_CUBIC), model_name))
        reshapedimageinput = np.asarray(reshapedimageinput)


    # --------------------------------------------------------------------------------------------------------------------------------------


    variables_to_restore = tf.contrib.framework.get_variables_to_restore()

    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        if(flag):
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
            init_fn(sess)
        else:
            # init = tf.global_variables_initializer()
            # sess.run(init)
            net.load(model_path, sess)
        feed = {images: reshapedimageinput}
        a = sess.run(layer_features, feed_dict=feed)
        # print(a.shape)
        return a

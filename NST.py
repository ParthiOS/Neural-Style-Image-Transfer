import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numba 
from numba import cuda
import tensorflow as tf
import pprint


import numpy as np 

# debugging code for GPU detection , currently not working
print(sys.executable)

print(tf.__version__)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))




tf.random.set_seed(272) # DO NOT CHANGE otherwise you will get different results each time/ hard totest
pp = pprint.PrettyPrinter(indent=4)
img_size = 400

# PLEASE NOTE: the file in weights= section is actually just a file downloaded us# theing imageNet's pre existing data base,
# The initial code to download this file was temp_file = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# then temp_file.save_weights('vgg19_weights.h5') was called to save the file to the working dir
temp_file = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
temp_file.save_weights('vgg19_weights.h5')
vgg = tf.keras.applications.VGG19(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                weights='vgg19_weights.h5'
)

vgg.trainable = False
pp.pprint(vgg) # check 
# this is the content image
# temporarily deativate
#content_image = Image.open("")
#content_image

def compute_content_cost(content_output, generated_output):
    
    # a_C and a_G are tensors
    a_C = content_output[-1]
    a_G = generated_output[-1]

    # dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape 'a_C' and 'a_G'
    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])
    
    # calc the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4.0 * n_H * n_W * n_C)
    
    
    
    return J_content



# this



def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """  
    
    
    # Multiply A with its transpose 
    GA = tf.matmul(A, tf.transpose(A))
    
    

    return GA



def compute_layer_style_cost(a_S, a_G):
    
    # a_S and a_G are tensors
    
    # dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the tensors : (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_S = tf.transpose(a_S)
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    a_G = tf.transpose(a_G)

    # gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    factor = (1 / (4 * n_C**2 * (n_H * n_W)**2))
    J_style_layer = factor * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    
    
    return J_style_layer

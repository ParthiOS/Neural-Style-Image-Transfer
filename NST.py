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

# debugging code for GPU detection 
print(sys.executable)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.random.set_seed(272) # DO NOT CHANGE THIS VALUE
pp = pprint.PrettyPrinter(indent=4)
img_size = 800

# PLEASE NOTE: the file in weights= section is actually just a file downloaded us# theing imageNet's pre existing data base,
# The initial code to download this file was temp_file = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# then model.save_weights('vgg19_weights.h5') was called to save the file to the working dir
temp_file = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg = tf.keras.applications.VGG19(include_top=False,
                                input_shape=(img_size, img_size, 3),
                                weights='pretrained-model/vgg19_weights.h5'
)

vgg.trainable = False
pp.pprint(vgg)
# this is the content image
#image = tf.image.resize(image, (img_size, img_size))
#content_image = Image.open("C:/Users/Parth/OneDrive/Desktop/Projects/Neural Style Image Transfer/Files/tf/W4A2/images/louvre.jpg")
#content_image

def compute_content_cost(content_output, generated_output):
    
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape 'a_C' and 'a_G' which are the tensors of dim: (1, n_H, n_W, n_C) and contain the values produced by the neural network
    # Here we unroll them to go from the 3D array to 2D
    a_C_unrolled = tf.reshape(a_C, [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, n_H * n_W, n_C])
    
    # compute the cost with tensorflow this is the formula for content cost
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4.0 * n_H * n_W * n_C)
    
    
    
    return J_content

#print(compute_content_cost_test(compute_content_cost))

# this is the example style image

example = Image.open("C:/Users/Parth/OneDrive/Desktop/Projects/Neural Style Image Transfer/Files/tf/W4A2/images/drop-of-water.jpg")
example

def gram_matrix(A):
    
    # Multiply A with its transpose (≈1 line)
    GA = tf.matmul(A, tf.transpose(A))
    
   

    return GA

#print(gram_matrix_test(gram_matrix))

def compute_layer_style_cost(a_S, a_G):
    
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_S = tf.transpose(a_S)
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    a_G = tf.transpose(a_G)

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    factor = (1 / (4 * n_C**2 * (n_H * n_W)**2))
    J_style_layer = factor * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    
    
    return J_style_layer


#compute_layer_style_cost_test(compute_layer_style_cost)

for layer in vgg.layers:
    print(layer.name)

vgg.get_layer('block5_conv4').output

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    
    
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style

# higher alpha values will make generated image resemble the content picture more
#  and the higher betas will be for more relation to style
@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=35):
    
    # Weighted sum of content and style costs
    J = alpha * J_content + beta * J_style
    
   

    return J

#total_cost(total_cost)


# These are the real content and style images that will be used in the calculations

content_image = np.array(Image.open("C:/Users/parth/OneDrive/Desktop/Projects/Neural Style Image Transfer/Files/tf/W4A2/images/cityscape1.jpg").resize((img_size, img_size)))

content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

print(content_image.shape)
imshow(content_image[0])
plt.show()

style_image =  np.array(Image.open("C:/Users/Parth/OneDrive/Desktop/Projects/Neural Style Image Transfer/Files/tf/W4A2/images/japanese.jpg").resize((img_size, img_size)))

style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

print(style_image.shape)
imshow(style_image[0])
plt.show()


generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()


def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

content_layer = [('block5_conv4', 1)]

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder

# preprocess the images to find a_C and  a_S

preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

def clip_0_1(image):
    
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)  #  vgg_model_outputs is the function to get the outputs of VGG model
        
        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G)  
        
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)  

        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=20, beta=25)
        
    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    
    # For grading purposes
    return J

        


generated_image = tf.Variable(generated_image)

# this trains the model by applying the iteration on the generated image a total of epochs 
epochs = 20000
for i in range(epochs):
    train_step(generated_image)
    print(i)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"C:/Users/Parth/OneDrive/Desktop/Projects/Neural Style Image Transfer/Files/tf/W4A2/output/image_{i}.jpg")
        






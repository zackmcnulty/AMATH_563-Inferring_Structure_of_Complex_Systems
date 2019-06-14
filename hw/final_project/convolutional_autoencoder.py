'''
In this rendition, we will be using convolutional neural networks to downsize the images.
Since these types of neural nets are better at storing the local structure of the
data they are compressing, they should work better than a standard feedforward layer
for encoding/decoding. 

'''

# This first line imports a bunch of different neuron layers we can use
# Input: self-explanatory; stores input data that is fed-forward to future layers

# Dense: standard feedforward layer with output = activation_func(Weights*input + bias)

# Conv2D: convolutional layer; each neuron in this layer has a receptive field that it looks at;
#         unlike a typical feedforward layer, these neurons only have connections to the neurons 
#         in their receptive field (some subset of the previous layer). For images, this is typically
#         just set to be some square (i.e. 3 by 3 grid of pixels) in the image.
#         For each receptive field, we define a number of neurons that are attached to it called the
#         DEPTH of the CNN. All the neurons at a given depth form what is called a "filter". The idea
#         is that we collect this series of filters where each filter acts on the entire image by
#         combining the results of a bunch of local analysis of the image. Each filter extracts its
#         own set of features.
#         Shared Weights: the neurons at each depth use the same weights/bias

# maxPooling: this neuron is what actually allows for the downsampling, reducing the layers size, not Conv2D.
#             here, each neuron in this layer is connected to some subset of the previous layer
#             It chooses its activity based on the max activity in the previous layer. Typically these are
#             chosen to be a small grid for the same reasons as in Conv2D: to capture local properties of the image
#             NOTE: these grids/filters typically don't overlap?

# UpSampling: allows network to regain the dimensionality it lost during maxPooling. For each neuron in the previous
#             layer, it simply generates multiple copies of neurons matching that activity level?

# SimpleRNN: a fully connected recurrent neural network. Output of network is fed back in as input (with
#            a weights matrix)
# Reshape: Layer that reshapes inputs to desired shape
# ZeroPadding2D: pads input with zeros (so its dimensions are divisible by 2^k for reduction by max pooling)
# Cropping2D: undos this ^ by removing padding
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, SimpleRNN, Reshape, ZeroPadding2D, Cropping2D
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras import backend as K
import os
import sys
from pathlib import Path
import numpy as np
import cv2 # for reading videos
from matplotlib import pyplot as plt
import argparse
import time

# note now we are not going to vectorize our data because we care about the local structure
# instead, input will be fed in as matrices (i.e. frame by frame)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--name', default='conv_autoencoder_LSTM_BCE')
parser.add_argument('--load') # specify a file to load
parser.add_argument('--l1', default=0, type=float, help='l1 penalization on kernels of convolutional layers') # specify a file to load
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_name = args.name
l1 = args.l1
num_results_shown = 10 # number of reconstructed frames vs original to show on test set

# Save the Keras model
if args.load is not None and args.name is None:
    model_filename = Path(args.load)
else:
    model_filename = Path(model_name + "_l1_" + str(l1) + ".h5")
    model_filename = 'models' / model_filename 

if model_filename.exists():
    answer = input('File name ' + model_filename.name + ' already exists. Overwrite [y/n]? ')
    if not 'y' in answer.lower():
        sys.exit()

# Load movie clips ===============================================================================`

num_train_movies = 0
num_test_movies = 0

# count movies first so I can pre-allocate space in numpy arrays
for i, f in enumerate(os.scandir('./movie_files')):
    if f.name.startswith('train'):
        num_train_movies += 1
    elif f.name.startswith('test'):
        num_test_movies += 1

    # should be the same for all movies
    if i == 0:  
        cap = cv2.VideoCapture(f.path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

image_shape = [frameHeight, frameWidth]
    
x_train = np.empty((frameCount * num_train_movies, frameHeight, frameWidth, 1))
x_test = np.empty((frameCount * num_test_movies, frameHeight, frameWidth, 1))

train_ind = 0
test_ind = 0
for f in os.scandir('./movie_files'):
    cap = cv2.VideoCapture(f.path)
    while True: 
        # ret is a boolean that captures whether or not the frame was read properly (i.e.
        # whether or not we have reached end of video and run out of frames to read)
        ret , frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if f.name.startswith('train'):
            x_train[train_ind, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            train_ind += 1
        elif f.name.startswith('test'):
            x_test[test_ind, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            test_ind += 1

#plt.imshow(x_test[0].reshape((frameHeight,frameWidth)))
#plt.show()




# convert the image into a lower dimensional representation

# Conv2D( depth, kernel/filter size, ...)
# The depth refers to the number of neurons connected to the same receptive field in the input.
# In the case below, we have 16 neurons connected to each 3 x 3 receptive field of pixels in 
# the input. These receptive fields are generated by moving the 3 x 3 filter one pixel at a time 
# (there is another parameter called 'strides' that can specify how many pixels to move each time).
# These 16 neurons convolve with their receptive field to capture the local patterns. Important
# in cases where local structure is the most important. As these layers only need to consider their
# receptive field, they are NOT connected to any other neurons in the previous layer

# To the right, we have shown how the dimensionality of the input changes. 

# NOTE: maxpooling does NOT effect the depth of the layers (number of filters); it only reduces the number
# of receptive fields.

# Maxpooling reduces the dimensionality


#Define Neural Network Structure ===============================================================================`
if args.load is not None:
    # do stuff
    model = load_model(args.load)
else:
    image_shape.append(1)

    model = Sequential()

    # padding prevents the change of shape due to down/upsampling
    model.add(ZeroPadding2D(((2,1),(2,1)), input_shape = image_shape))  # (61,61, 1) --> (64,64, 1)

    model.add(Conv2D(16, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(l1)))         # (64, 64, 1) --> (64, 64, 16) 
    model.add(MaxPooling2D((2,2), padding='same'))                          # (64, 64, 16) --> (32, 32, 16)
    model.add(Conv2D(8, (3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1)))      # (32,32, 16) --> (32,32, 8)
    model.add(MaxPooling2D((2,2), padding = 'same'))                        # (32, 32, 8) --> (16,16, 8)
    model.add(Conv2D(8, (3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1)))      # (16,16,8) --> (16,16,4)
    model.add(MaxPooling2D((2,2), padding = 'same'))                        # (16, 16, 4) --> (4, 4, 4)
    model.add(Conv2D(8, (3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1)))      # (16,16,8) --> (16,16,4)
    model.add(MaxPooling2D((2,2), padding = 'same'))                        # (16, 16, 4) --> (4, 4, 4)
    model.add(Conv2D(4, (3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1)))      # (4,4,4) --> (4,4,1)


    # decode the lower dimensional representation back into an image
    model.add(Conv2D(4,(3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1))) # (4,4,1) --> (4,4,4)
    model.add(UpSampling2D((2,2)))                                    # (4,4,4) -> (16,16,4)
    model.add(Conv2D(8,(3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1))) # (4,4,1) --> (4,4,4)
    model.add(UpSampling2D((2,2)))                                    # (4,4,4) -> (16,16,4)
    model.add(Conv2D(8,(3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1(l1))) # (16,16,4) --> (16,16,8)
    model.add(UpSampling2D((2,2)))                                    # (16,16,8) -> (32,32,8)
    model.add(Conv2D(16, (3,3), activation = 'relu', padding='same', kernel_regularizer=regularizers.l1(l1)))# (32,32,8) -> (32,32,16)
    model.add(UpSampling2D((2,2)))                                    # (32,32,16) -> (64,64,16)
    # want to use sigmoid as our final activation function to make the output more
    model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l1(l1))) # (64,64,16) -> (64,64, 1)

    model.add(Cropping2D(((2,1),(2,1)))) # (64,64,1) -> (61,61, 1)

    model.compile(optimizer = 'adam', loss='binary_crossentropy')

    #for layer in model.layers:
    #    print(layer.output_shape)

# ================================================================================================
# fit model (train network)!
model.fit(x_train, x_train,
          epochs = epochs, 
          batch_size = batch_size, 
          shuffle = True,
          validation_data = (x_test, x_test))

# NOTE: Saves the model to the given model name in the folder ./models
model.save(str(model_filename))
model.summary()

# make predictions!
predicted_images = model.predict(x_test)
true_images = x_test

# plot stuff ====================================================================================

n = num_results_shown # number of digits to display

plt.figure(figsize=(20,4))
for i in range(n):

    # display original (in top row)
    ax = plt.subplot(2, n, i+1) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(true_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False

    # display predicted (in bottom row)
    ax = plt.subplot(2, n, i+ 1 + n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(predicted_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False

plt.show()


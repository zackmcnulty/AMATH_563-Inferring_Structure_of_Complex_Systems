'''
The task is to build an autoencoder for the MNIST database. This will learn
to form a compressed representation of the handwritten digit (encoding)
and then decode it back into the original image.

Here, I follow the tutorial found at:

    https://blog.keras.io/building-autoencoders-in-keras.html

In this rendition, we will be using convolutional neural networks to downsize the images.
Since these types of neural nets are better at storing the local structure of the
data they are compression, they should work better than a standard feedforward layer
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
# TimeDistributed: applies this layer to every element in the provided sequence.
# ConvLSTM2D: reccurrent neural network (LSTM) that is applied convolutionally
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, SimpleRNN, Reshape, ZeroPadding2D, Cropping2D, TimeDistributed, Flatten, ConvLSTM2D
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
parser.add_argument('--epochs', default=50, type=int, help='number of epochs in neural net training')
parser.add_argument('--batch_size', default=5, type=int, help='batch size for training')
parser.add_argument('--name', default='rnn_predictior_BCE', help='file name to save to RNN Keras model under in the ./models folder')
parser.add_argument('--load', help='file name for a previously trained RNN that you wish to train further.') # specify a full pre-trained RNN model to load
parser.add_argument('--autoencoder', help='filename that stores the Keras model with the pre-trained convolutional autoencoder (see convolution_autoencoder.py)') # specify the filename for the weights trained to autoencode
parser.add_argument('--dt', default=1, type=int, help='Number of frames ahead in movie to make prediction')
parser.add_argument('--l1', default=0, type=float, help='lambda value for l1 regularization on RNN weights')
args = parser.parse_args()

if args.load is None and args.autoencoder is None:
    raise ValueError('Either a autoencoder model file must be specified (via --autoencoder filename) or a pre-trained model (via --load)')

epochs = args.epochs
batch_size = args.batch_size
model_name = args.name
num_results_shown = 10 # number of reconstructed frames vs original to show on test set
dt = args.dt # number of frames ahead to make a prediction
l1 = args.l1 # l1 penalization on RNN weights

# For simple RNN
rnn_size = 64

# Check if a model already exists with the given filename
if not args.load is None and args.name is None:
    model_filename = Path(args.load)
else:
    model_filename = Path(model_name + "_dt_" + str(dt) +'_l1_' + str(l1) +  ".h5")
    model_filename = 'models' / model_filename

if model_filename.exists():
    answer = input('File name ' + model_filename.name + ' already exists. Overwrite [y/n]? ')
    if not 'y' in answer.lower():
        sys.exit()

# Load movie clips ===============================================================================`

# NOTE: Unlike in the convolutional_autoencoder case where we just stacked all the frames to form our 
# inputs, we want to retain this data as sequences so we can do sequence prediction with our RNN (i.e. predict
# the future frames given a set of current frames). Thus,
# our inputs will be stored as full films and the outputs will be staggered in time
# Thus, inputs to neural net will be of the form (# films, # frames/film, frame height, frame width)
# with each single input as (1, # frames, frame height, frame width)

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
    
# you start with frameCount frames. Since the input and output have to be staggered by dt, this means
# that the input and output can have at most (frameCount - dt) frames as dt of their frames do not overlap.
num_frames = frameCount - dt

x_train = np.empty((num_train_movies, num_frames, frameHeight, frameWidth, 1))
y_train = np.empty((num_train_movies, num_frames, frameHeight, frameWidth, 1))
x_test = np.empty((num_test_movies, num_frames, frameHeight, frameWidth, 1))
y_test = np.empty((num_test_movies, num_frames, frameHeight, frameWidth, 1))

train_ind = 0
train_movie_num = 0
test_ind = 0
test_movie_num = 0
for f in os.scandir('./movie_files'):

    cap = cv2.VideoCapture(f.path)
    while True: 
        # ret is a boolean that captures whether or not the frame was read properly (i.e.
        # whether or not we have reached end of video and run out of frames to read)
        ret , frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if f.name.startswith('train'):
            if train_ind < num_frames:  
                x_train[train_movie_num, train_ind, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            if train_ind >= dt:
                y_train[train_movie_num, train_ind - dt, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]

            train_ind += 1

        elif f.name.startswith('test'):
            if test_ind < num_frames:  
                x_test[test_movie_num, test_ind, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            if test_ind >= dt:
                y_test[test_movie_num, test_ind - dt, :, :, 0] = gray / 255.0 # NOTE: convert pixels to [0,1]
            test_ind += 1

    if f.name.startswith('train'): 
        train_movie_num += 1
        train_ind = 0
    elif f.name.startswith('test'): 
        test_movie_num += 1
        test_ind =  0

# TODO: Delete this
#x_train = x_train[:, :, :, :, 0]
#x_test = x_test[:, :, :, :, 0]

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
    autoencoder = load_model(args.autoencoder)

    # Since the autoencoder is symmetric (the encoder section has exactly the same number of layers
    # as the decoder region) we can just place the RNN in the middle:
    # We also have to convert all the layers to TimeDistributed so they can handle sequence prediction, but this
    # does not effect their weights. All the non-RNN layers will just apply their weights to each element in the
    # sequence individually, but the full sequence will be fed into (and come out of) the RNN
    num_layers = len(autoencoder.layers)

    model = Sequential()
    for i in range(num_layers // 2):
        model.add(TimeDistributed(autoencoder.layers[i]))
    

    out_shape = autoencoder.layers[num_layers//2 - 1].output_shape


    #SimpleRNN
    model.add(TimeDistributed(Flatten()))
    model.add(SimpleRNN(rnn_size, 
                  return_sequences = True,
                  activation = 'tanh',
                  recurrent_regularizer=regularizers.l1(l1),
                  name='rnn'))
    # NOTE: since the RNN changes size of output of the final Conv2D layer in encoding section, we somehow have
    #       to map the dimension back down. This is what the Dense layer below does
    #model.add(TimeDistributed(Dense(out_shape[1] * out_shape[2], activation = 'relu', name = 'ff')))
    model.add(TimeDistributed(Reshape((out_shape[1], out_shape[2], out_shape[3]))))

    for i in range(num_layers//2, num_layers):
        model.add(TimeDistributed(autoencoder.layers[i]))

    # set non-reccurent layers to untrainable; we already trained these to be autoencoders, so the RNN
    # just has to learn how to move the object in the low dimensional space
    for layer in model.layers:
        if not (layer.name == 'rnn' or layer.name == 'ff'):
            layer.trainable = False

    model.compile(optimizer = 'adam', loss='binary_crossentropy')

# ================================================================================================
# fit model (train network)!
model.fit(x_train, y_train,
          epochs = epochs, 
          batch_size = batch_size, 
          shuffle = True,
          validation_data = (x_test, y_test))
#          validation_split = 0.1)


# NOTE: Saves the model to the given model name in the folder ./models
model.save(str(model_filename))
model.summary(line_length=100)

# make predictions on training dataset!
predicted_images = model.predict(x_train)[0] # the [0] just takes predictions for first video
true_images = y_train[0, :, :, :, :]
initial_images = x_train[0, :, :, :, :]

# plot stuff ====================================================================================

n = num_results_shown # number of frames to display

plt.figure(figsize=(20,4))
for i in range(n):
    # display original at time t (in top row)
    ax = plt.subplot(3, n, i+1) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(initial_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t)')

    # display original at time t+dt (in middle row)
    ax = plt.subplot(3, n, i+ 1 + n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(true_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t+dt)')

    # display predicted at time t+dt (in bottom row)
    ax = plt.subplot(3, n, i+ 1 + 2*n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(predicted_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Predicted (t+dt)')

plt.show()






# make predictions on test dataset!
predicted_images = model.predict(x_test)[0] # the [0] just takes predictions for first video
true_images = y_test[0, :, :, :, :]
initial_images = x_test[0, :, :, :, :]

# plot stuff ====================================================================================

n = num_results_shown # number of frames to display

plt.figure(figsize=(20,4))
for i in range(n):
    # display original at time t (in top row)
    ax = plt.subplot(3, n, i+1) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(initial_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t)')

    # display original at time t+dt (in middle row)
    ax = plt.subplot(3, n, i+ 1 + n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(true_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Original (t+dt)')

    # display predicted at time t+dt (in bottom row)
    ax = plt.subplot(3, n, i+ 1 + 2*n) # which subplot to work with; 2 rows, n columns, slot i+1
    plt.imshow(predicted_images[i].reshape(image_shape[0], image_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible = False
    ax.get_yaxis().set_visible = False
    if i == 0:
        plt.ylabel('Predicted (t+dt)')

plt.show()


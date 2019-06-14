# AMATH 563 Final Project



## File Descriptions and Uses

#### dynamical_systems.py

This file stores the systems of differential equations that define the 
motion of objects throughout the movie. It is referenced in the make_movies.py file when defining objects to be add to the video.


#### get_trajectory.py

Peforms the numerical integration for a given set of objects and the equations
defining their motion. Normalizes this motion so that all objects move within a [0,1] x [0,1] window.

#### graphics.py

**NOT MY CODE**
Simple graphics package written for python. The source code can be found [here](https://mcsp.wartburg.edu/zelle/python/graphics.py).

#### make_movie.py

Generates the movie files used for training, testing, and data analysis. This file is called by the generate_training_data.sh and
generate_uniform.sh scripts to make a batch of movie files simultaneously.

usage: python3 make_movie.py

optional flags:
-name : name to give to the movie file
--folder : where to save the movie file
--theta : generate a movie with the given angle of oscillation
-random : generate a movie with a random angle of oscillation
-p_test : percent of angles to make the test dataset



#### convolutional_autoencoder.py

Trains the autoencoder on the give set of movies located in movie_files. Builds a training dataset using the movie files labeled "train" 
and a testing dataset using the movie files labeled "test"

usage:  python3 convolutional_autoencoder.py

optional flags
--epochs : number of epochs to train for
--batch_size : batch size to train with using stochastic gradient descent
--name : name to give to the keras model generated after training
--load : name of a previous keras model to load; train on top of the current model rather than starting fresh
--l1 : L1 penalty to add to weights in network

#### rnn_predictor.py

Trains the RNN on the given set of movies in the movie_files/ folder using a pre-trained autoencoder.

usage: python3 rnn_predictor.py 

optional flags
--epochs : number of epochs to train for
--batch_size : batch size to train with using stochastic gradient descent
--name : name to give to the keras model generated after training
--load : name of a previous RNN (keras model)  to load; train on top of the current model rather than starting fresh
--autoencoder : name of an autoencoder (keras model) to add the RNN to.
--l1 : L1 penalty to add to weights in network
--dt : number of frames ahead you want the RNN to predict

#### movie_files

Movie files used for training/validation of the RNN and autoencoder

#### test_movies

Movie files used as part of analysis.

#### generate_training_data.sh

Generates a specified number of movie files by calling make_movie.py. Creates a testing and training dataset with disjoint
angles of oscillations.

usage: ./generate_training_data.sh

#### generate_uniform.sh

Generates a specified number of movie files by calling make_movie.py. Uniformly distributes the angles of oscillation between
two angles specified in the file.

usage: ./generate_uniform.sh

#### analysis.py

All analysis of movie files occurs in this file (PCA, DMD, tuning curves, plotting trajectories in PC space, etc)
Studies the neural representations of both the RNN and the encoder (input to the RNN) by loading a keras RNN + autoencoder model
and testing it on a provided set of test movie files.

usage: python3 analysis.py -load /path/to/RNN_keras_model -movie_folder /path/to/folder_of_movies_for_analysis

optional flags:
--svm : run svm on encoder and RNN neural representations
--svd : run SVD on encoder/RNN neural representations
--uniform : plot tuning curves of neural representations with respect to angle of oscillation


# AMATH 563 Final Project



## File Descriptions and Uses

#### dynamical_systems.py

This file stores the systems of differential equations that define the 
motion of objects throughout the movie.

#### get_trajectory.py

Peforms the numerical integration for a given set of objects and the equations
defining their motion. Normalizes this motion so that all objects move within a [0,1] x [0,1] window.

#### graphics.py

**NOT MY CODE**
Simple graphics package written for python. The source code can be found [here](https://mcsp.wartburg.edu/zelle/python/graphics.py).

#### make_movie.py


#### convolutional_autoencoder.py

Trains the autoencoder on the give set of movies.

#### rnn_predictor.py

Trains the RNN on the given set of movies using a pre-trained autoencoder.

#### movie_files

Movie files used for training/validation of the RNN and autoencoder

#### test_movies

Movie files used as part of analysis.

#### generate_training_data.sh

Generates a specified number of movie files by calling make_movie.py

#### generate_uniform.sh
#### analysis.py

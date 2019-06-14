#!/bin/bash

# This file will run the make_movies file many times (exact number chosen below) to generate testing and
# training data. I used this file to generate a bunch of videos of the angled spring at various angles.
# The training dataset includes angles between pi/6 and 5pi/6 (and by symmetry 7pi/6 to 11pi/6) and
# the testing dataset includes the angles between -pi/6 and pi/6. These angles between testing and training
# are chosen to be completely DISJOINT.

# usage:
# ./generate_data.sh

# -p_test = in [0,1];  percent of the directions (i.e. 0, 2pi) that are used for testing data. i.e. if 1/6 is provided
#		then angles -pi/6 to pi/6 (for the angled spring system) are used to make testing movies and angles
#		pi/6 to 5pi/6 are used for training movies. This divides all possible angles into disjoint train/test (due to symmetry)
# -train = generate a movie for the training dataset (leaving out this flag generates a movie for test set)
# -random = This is just a flag telling the script I am running it from an external file rather than specifying the objects
#	    in the file itself.


num_train_movies=250
num_test_movies=50
percent_test=0.2

# generate training movies
for i in $( seq $num_train_movies)
do
       filename=`printf train%04d $i`
       python3 make_movie.py -name $filename -random -train -p_test $percent_test	
done

for i in $( seq $num_test_movies)
do
       filename=`printf test%04d $i`
       python3 make_movie.py -name $filename -random -p_test $percent_test	
done

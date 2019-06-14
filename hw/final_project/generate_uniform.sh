#!/bin/bash

# This file will run the make_movies file many times (exact number chosen below) to generate movies for analysis
# I used this file to generate a bunch of videos of the angled spring at various angles uniformly chosen along an angle.

# usage:
# ./generate_uniform.sh


start_angle=160
stop_angle=200
num_movies=40
# number of movies to make with angles uniformly between [start, stop]
folder="left_right" # foldername in ./test_movies to save movies



diff=$(expr $stop_angle - $start_angle)
step_size=$(expr $diff / $num_movies)

#touch './test_movies/uniform/labels.csv'

# generate training movies
for i in $( seq $num_movies)
do
	angle=$(($start_angle + $(($i * $step_size))))
       filename=`printf uniform%03d $angle`
       python3 make_movie.py -name $filename --theta $angle --folder "./test_movies/$folder"

       # add the given label to the labels file
       printf $angle >> "./test_movies/$folder/labels.csv"
       printf ',' >> "./test_movies/$folder/labels.csv"
done

printf '%s\n' '$' 's/.$//' wq | ex './test_movies/uniform/labels.csv' 

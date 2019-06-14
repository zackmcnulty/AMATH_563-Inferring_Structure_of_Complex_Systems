'''
make_movie.py

Given a series of objects that are influenced by some dynamical system, creates a movie (frame by frame) of those
objects moving (and possibly rotating) throughout that dynamical system.

Creates a movie file in the directory './movie_files/move_file_name.mpeg4' within the current working directory.

Usage:

    python3 make_movie.py movie_filename [, frames_per_second = 10]

Dynamics are adjusted to fit the graphics window so to avoid distorting dynamics its best to set the window
height and width to be equal.

'''

# NOTE: To support rotation, all objects will have to be regular polygons. We can then calculate rotation
#       manually and redraw the polygon
# see: https://stackoverflow.com/questions/45508202/how-to-rotate-a-polygon

from graphics import * # allows for drawing
import get_trajectory # calculates the trajectory of an object given a dynamical system
import dynamical_systems as ds # a few function definitions for common dynamical systems
import numpy as np
import time
from PIL import Image as NewImage
import os
import sys
import argparse
from pathlib import Path



# Define objects and their motion  ==============================================================================================

parser = argparse.ArgumentParser()
parser.add_argument('-random', action='store_true')
parser.add_argument('-name', required=True)
parser.add_argument('--folder', default='./movie_files')
parser.add_argument('-train', action='store_true', help='add to training dataset')
parser.add_argument('-p_test', type=float, help='percent of [0, 2pi] to allocate to testing dataset') 
parser.add_argument('--theta', type=float, help='angle of oscillation for movie')
args = parser.parse_args()

# each object will be specified by (motion = (dynamics, initial_condition), shape)
# Shapes will be specified as graphics objects
# NOTE: The depth of objects in the image is specified by the order they appear in "objects" below.
#       Shapes earlier in "objects" appear deeper in the image



# Randomly generate training/testing data
if args.random: # ... for generating large batches of movies; uses .sh script
    #NOTE: See the file generate_data.sh for more information about
    # what I used this 'else' section for

    args.p_test = args.p_test / 2
    low_train = np.pi * args.p_test
    high_train = np.pi - low_train
    low_test = -1 * low_train
    high_test = low_train

    if args.train:
        theta = np.random.uniform(low=low_train, high=high_train)
    else:
        theta = np.random.uniform(low=low_test, high=high_test)

    objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=theta), )  
    shapes =  (Rectangle(Point(-3,-3), Point(3,3)), )
    fill_colors = ['black']
    outline_colors = ['black']



# generate movies with angles uniformly distributed along some range.
elif args.theta is not None:
    
    # convert angle given in degrees to radians
    args.theta = np.pi / 180 * args.theta

    objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=args.theta), )  
    shapes =  (Rectangle(Point(-3,-3), Point(3,3)), )
    fill_colors = ['black']
    outline_colors = ['black']


else: # specify the objects in this program 
    objects = (ds.f_angled_spring(initial_condition=[0,0,1], theta=np.pi/4 ), )  

    # NOTE: set all shapes to start at centered (0,0)
    shapes =  (Rectangle(Point(-5,-5), Point(5,5)), )

    # NOTE: Colors can be strings ('black', 'blue', 'green', 'black', 'yellow') or RBG in the form of a tuple (r, g, b)

    # Randomly choose colors for the objects.
    fill_colors = [np.random.choice(list(range(256)), 3) for i in range(len(objects))]
    outline_colors = [np.random.choice(list(range(256)), 3) for i in range(len(objects))]




if len(objects) != len(shapes) or len(objects) != len(fill_colors) or len(objects) != len(outline_colors):
    raise ValueError("The number of objects, shapes, and colors specified must all match.")



# Define Properties of Graphics window (Video Properties) ===================================================================
     

# Number of frames to generate for the movie and the frames per second for the movie respectively.
num_frames = 35
movie_fps = 5

# Specify properties of the graphics window; Below allows you to control the size in pixels of
# the window (and hence the resolution in the video)
# The padding feature generates a an area of white space above/below and to the left/right of the window.
# I found this helpful for keeping the objects from running off the edge of the screen. As their motion
# moves around their center so if the object is too big it will go over the edge.
win_height = 50
win_width = 50 
padding = 5















# ===============================================================================================================================
# Code below can remain unchanged. For implementation details only. All details of video specified above.
# ===============================================================================================================================

def save_frame(win, frame):
    '''
    Saves as an image the current graphics window. This will generate a single
    frame of our video.
    '''
    # saves the current TKinter object in postscript format
    win.postscript(file="tmp_frames/image.eps", colormode='color')

    # Convert from eps format to gif format using PIL
    img = NewImage.open("tmp_frames/image.eps")
    img.save('tmp_images/frame{:05d}.png'.format(frame), "png")




# check if the movie filename already exists to avoiding overwriting files.
movie_filename = args.name
full_path = Path(args.folder) / (movie_filename + '.mp4')
if not os.path.exists(args.folder):
   os.system('mkdir ' + args.folder) 
elif os.path.exists(full_path):
    answer = input("Filename " + str(full_path.absolute()) + " already exists. Overwrite? [y/n] : ")
    if 'n' in answer.lower():
        sys.exit(0)

# create temporary directories to save the frames/images that are generated
try:
    # clear folders if they are there from a previous stopped run
    os.system('rm -rf tmp_images')
    os.system('rm -rf tmp_frames')
except:
    pass

os.system("mkdir tmp_images/")
os.system("mkdir tmp_frames/")




# Calculate object trajectories
(t_vals, x_vals, y_vals) = get_trajectory.get_trajectories(objects, num_steps=num_frames)

win = GraphWin( width = win_width + 2*padding, height=win_height + 2*padding)


# draw the initial state and color the objects 
for i, shape in enumerate(shapes):
    x = x_vals[i, 0]
    y = y_vals[i, 0]

    # move our object to its initial condition + padding. Since our coordinates from
    # get trajectories are normalized to [0,1] we can find the location in the window
    # simply by multiplying by window height.
    shape._move(padding + x * win_width, padding + y * win_height)
    shape.draw(win)

    fill_color = fill_colors[i]
    outline_color = outline_colors[i]
    if type(fill_color) != str:
        fill_color = color_rgb(fill_color[0],fill_color[1],fill_color[2]) # color_rgb is a function from graphics.py

    if type(outline_color) != str:
        outline_color = color_rgb(outline_color[0],outline_color[1],outline_color[2]) # color_rgb is a function from graphics.py

    shape.setFill(fill_color)
    shape.setOutline(outline_color)

    save_frame(win, 1)


for frame in range(1, len(t_vals)):
    for i, shape in enumerate(shapes):
        # Since our dynamics are normalized in get_trajectories, we can just multiply them by the window
        # size to adjust this motion to the graphics window
        dx = win_width * (x_vals[i, frame] - x_vals[i, frame-1]) 
        dy = win_height * (y_vals[i, frame] - y_vals[i, frame - 1])
        shape._move(dx, dy)

    win.redraw()
    save_frame(win, frame + 1)


# Convert all these frames into a movie
os.system("ffmpeg -r {:d} -i ./tmp_images/frame%05d.png -vcodec mpeg4 -y {:s}".format(movie_fps, str(full_path.absolute())))

# delete the temporary folders created
os.system('rm -rf tmp_images')
os.system('rm -rf tmp_frames')



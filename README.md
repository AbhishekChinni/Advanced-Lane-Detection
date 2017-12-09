# Advanced-Lane-Detection

TODO: Detailed explanation

The project is aimed at lane detection from a video taken from a front-camera of a car to automatically detect lanes and determine the radius of curvature and the vehicle position in the lane. Implemented in OpenCV Python.

# Algorithm:
1) Fix Camera Distortion
2) Perspective Transformation
3) Lane detection with a combination of edge detection, R-Channel, S-Channel, L-Channel Thresholding.
4) Detecting the lane lines by histogram binning and box stacking.
5) Fitting a polynomial curve to the lane lines
6) Calculating the Radius of curvature and vehicular position in the lane.
7) Visualization

# Code: 
included in run.py

# Sample Output: https://www.youtube.com/watch?v=ORShZ8Ny0R0&feature=youtu.be

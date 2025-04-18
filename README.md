# Camera-based object tracking: A SORT-based algorithm

This code implements a [SORT](https://arxiv.org/abs/1602.00763)-based object tracking algorithm for monocular camera sensors. It has been tested against KITTI object tracking benchmark achieving a HOTA score of 58.

Example of application:

![](examples/0013_output_video.gif)

Tracker implementation can be found in `track_main.py` and scripts to run KITTI sequence start with prefix `run_*`

Further details can be found in the dedicated [blog post](https://blogjtr.com/posts/object-tracking-sort/).

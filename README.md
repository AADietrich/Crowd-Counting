# Crowd-Counting
Crowd Counting Project - Unofficial implementation of MRCnet using Keras

main.py is the entry-point for the program, un-comment function calls as necessary to run

params.py contains program-level configuration parameters
  Set the correct location of the input data directory here - file structure assumed to be that of Shanghaitech dataset



network.py contains all model code

preprocess.py contains most preprocessing code with helper methods in heatmap.py and crop.py

loading_shanghaitech notebook was used for iterative testing

All code in gpselle folder is from the original branch and is currently unused by the implementation



Crowd-counting project created by Aaron Dietrich as part of the BCIT GIS Program.

Project sponsored by Planetary Remote Sensing Inc. prscorp.ca

Shanghaitech dataset used for training and testing. Adaptive kernel heatmap generation implemented as described in (1)
Model architecture and hyperparameters taken from (2)

(1) Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, Yi Ma.2016.Single-Image Crowd Counting via Multi-Column Convolutional Neural Network.\url{https://zpascal.net/cvpr2016/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf}\\
(2) Bahmanyar, R., Vig, E., Reinartz, P. (2019). MRCNet: Crowd Counting and Density Map Estimation in Aerial and Ground Imagery. arXiv, (1909.12743). https://arxiv.org/abs/1909.12743

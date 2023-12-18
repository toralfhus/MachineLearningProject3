# MachineLearningProject3
The final exercise in the course FYS-STK4155

Group members:

* Frederic Eichenberger
* Toralf Husevåg
* Johan Blakkisrud

## Overview
This repository contains functions and outputs for project 3 of course applied data analysis and machine learning.

The task focus on using different tools to classify x-ray images into either normal, or diagnosed with pnemonia.

### CNN

A CNN is used for comparison, and the main functionality is found in the script CNN.py.

The data has to be in a specific structure for the dataloaders to work, and a preprocessesing function is found in the utility scripts.

The CNN uses transfer learning with the VCC16, that has to be downloaded beforehand, for example from here: https://www.kaggle.com/code/ligtfeather/x-ray-image-classification-using-pytorch

The number of epochs are set relatively low, but can be increased.

## Installation

The repository uses a set of library functions from different open sources, including

* sklearn
* seaborn
* pandas
* pytorch

Any compatible set of versions should work, but your milage may vary.

## License
Licenced under the WTFPL http://www.wtfpl.net/


## Files and Directories

Data from https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/overview

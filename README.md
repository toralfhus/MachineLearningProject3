# MachineLearningProject3
The final exercise in the course FYS-STK4155

Group members:

* Frederik Eichenberger
* Toralf Husevåg
* Johan Blakkisrud

## Overview
This repository contains functions and outputs for project 3 of course applied data analysis and machine learning.

The task focus on using different tools to classify x-ray images into either normal, or diagnosed with pnemonia.

### Hand crafted features
### Radiomics features with LR, RF, XGB, and a FFNN
1023 hand-crafted radiomic features were extracted using the pyRadiomics package. Using scikit-learn a logistic regression (LR), random forest (RF), and XGBoost (XGB) model was evaluated in addition to a dense feedforward neural network (FFNN) using tensorflow. Recursive feature elimination (RFE) was applied before training and testing the models, evaluated on average f1-score across a 3-fold cross validation applied to the training data.

### CNN

A CNN is used for comparison, and the main functionality is found in the script CNN.py.

The data has to be in a specific structure for the dataloaders to work, and a preprocessesing function is found in the utility scripts.

The CNN uses transfer learning with the VCC16, that has to be downloaded beforehand, for example from here: https://www.kaggle.com/code/ligtfeather/x-ray-image-classification-using-pytorch

The number of epochs are set relatively low, but can be increased.

For training, 20 percent of the already defined training data was used to validation during training.

## Installation

The repository uses a set of library functions from different open sources, including

* sklearn
* seaborn
* pandas
* pytorch
* pyradiomics
* tensorflow
 
Any compatible set of versions should work, but your milage may vary.

## License
Licenced under the WTFPL http://www.wtfpl.net/


## Files and Directories

Data set from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/

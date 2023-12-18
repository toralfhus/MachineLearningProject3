"""
Script to perform the CNN with transfer learning
Heavily inspired by https://www.kaggle.com/code/ligtfeather/x-ray-image-classification-using-pytorch

"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import skimage
from skimage import io, transform
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

EPOCHS = 5
data_dir = "rsna-pneumonia-detection-challenge/"
TEST = 'stage_2_test_images_preprocessed_hires_jpg'
TRAIN = 'stage_2_train_images_preprocessed_hires_jpg'
VAL ='stage_2_val_images_preprocessed_hires_jpg'

# Output values

name_of_model = "CNN_transfer_learning"

def data_transforms(phase):
    if phase == TRAIN:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
        
    if phase == VAL:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
    
    if phase == TEST:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])        
        
    return transform

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Set up the data loaders and data sets

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms(x)) 
                  for x in [TRAIN, VAL, TEST]}

dataloaders = {TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size = 32, shuffle=True), 
               VAL: torch.utils.data.DataLoader(image_datasets[VAL], batch_size = 1, shuffle=True), 
               TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size = 1, shuffle=True)}

# Identify the classes

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}
classes = image_datasets[TRAIN].classes
class_names = image_datasets[TRAIN].classes

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    """
    Function to train the model

    Parameters
    ----------
    model : pytorch model
        The model to be trained
    criterion : pytorch loss function
        The loss function to be used
    optimizer : pytorch optimizer
        The optimizer to be used
    scheduler : pytorch scheduler
        The scheduler to be used
    num_epochs : int
        The number of epochs to train for

    Returns
    -------
    model : pytorch model
        The trained model

    TODO: The saving of train and validation loss and accuracy should be 
          moved to a separate function or at least handled a bit less clunky


    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    training_loss_plotting = np.zeros(num_epochs)
    validation_loss_plotting = np.zeros(num_epochs)
    training_acc_plotting = np.zeros(num_epochs)
    validation_acc_plotting = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch+1, num_epochs))
        print("="*10)
        
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase==TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == TRAIN:
                training_loss_plotting[epoch] = epoch_loss
                training_acc_plotting = epoch_acc

            elif phase == VAL:
                validation_loss_plotting[epoch] = epoch_loss
                validation_acc_plotting[epoch] = epoch_acc


    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    try:

        np.save("train_loss_epoch.npy", training_loss_plotting)
        np.save("val_loss_epoch.npy", validation_loss_plotting)
        np.save("train_acc_epoch.npy", training_acc_plotting)
        np.save("val_acc_epoch.npy", validation_acc_plotting)

    except:

        print("Could not save the loss and accuracy arrays")
    
    return model

def test_model():
    """
    Function to test the model - this is done on the hold-out test set

    Returns
    -------
    true_labels : list
        List of true labels
    pred_labels : list
        List of predicted labels
    running_correct : float
        Number of correct predictions
    running_total : float
        Total number of predictions
    acc : float
        Accuracy of the model
    """
    running_correct = 0.0
    running_total = 0.0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data in dataloaders[TEST]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.append(labels.item())
            outputs = model_pre(inputs)
            _, preds = torch.max(outputs.data, 1)
            pred_labels.append(preds.item())
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()
        acc = running_correct/running_total
    return (true_labels, pred_labels, running_correct, running_total, acc)

# Load the pretrained model

model_pre = models.vgg16()
model_pre.load_state_dict(torch.load("vgg16-397923af.pth"))

# Print the model for inspection:

for param in model_pre.features.parameters():
    param.required_grad = False

num_features = model_pre.classifier[6].in_features
features = list(model_pre.classifier.children())[:-1] 
features.extend([nn.Linear(num_features, len(class_names))])
model_pre.classifier = nn.Sequential(*features) 
print(model_pre)

# Set the model to the device

model_pre = model_pre.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_pre.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
model_pre = train_model(model_pre, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)

# Save the model for posterity
torch.save({
    'model_state_dict': model_pre.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # You can include other elements as needed (e.g., epoch, loss history, etc.)
}, name_of_model + '.pth')


# Perform plotting - might crash

try:

    training_loss = np.load("train_loss_epoch.npy")
    validation_loss = np.load("val_loss_epoch.npy")
    training_acc =  np.load("train_acc_epoch.npy")
    validation_acc = np.load("val_acc_epoch.npy")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(training_loss, label = "Training loss")
    ax.plot(validation_loss, label = "Validation loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylim([None, 0.7])
    plt.legend()
    sns.despine()

    plt.savefig("Loss_CNN.png", dpi = 600)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(training_acc, label = "Training Accuracy")
    ax.plot(validation_acc, label = "Validation Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")

    ax.set_ylim([None, 0.9])
    plt.legend()
    sns.despine()

    plt.savefig("Accuracy_CNN.png", dpi = 600)

except:

    print("Could not plot the loss and accuracy arrays")

# Perform testing
    
true_labels, pred_labels, running_correct, running_total, acc = test_model()

# Print the accuracy
print("Total Correct: {}, Total Test Images: {}".format(running_correct, running_total))
print("Test Accuracy: ", acc)

# Plot the confusion matrix

cm = confusion_matrix(true_labels, pred_labels)
tn, fp, fn, tp = cm.ravel()
ax = sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("heatmap_cnn.png", dpi = 600)
    

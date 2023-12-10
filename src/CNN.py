
"""
Script to perform simple CNN classification of a dataset of x-ray images 
of healthy and pneumonia patients.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import csv
import pydicom
from skimage.transform import resize
import matplotlib.pyplot as plt
import imageio
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from tensorflow.keras.applications import ResNet50

from PIL import Image

# Do some checks on the training and test data against the labels, I do believe that
# the test data is not labelled?

#root_path_to_data = "../rsna-pneumonia-detection-challenge"
root_path_to_data = "rsna-pneumonia-detection-challenge"
path_to_save_cnn_model = "pneumonia_cnn.h5"

# Load the csv as a dict

def load_dicom_as_array(dicom_file):
    dicom_data = pydicom.read_file(dicom_file)
    image_array = dicom_data.pixel_array
    return image_array

def scrub_preprosessed_images(root_path_to_data):
    """
    Delete the content of the directories where the preprocessed images are stored.
    """

    train_images_dir = os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed')
    test_images_dir = os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed')

    train_images_dir_penum = os.path.join(train_images_dir, 'penum')
    train_images_dir_norm = os.path.join(train_images_dir, 'normal')

    test_images_dir_penum = os.path.join(test_images_dir, 'penum')
    test_images_dir_norm = os.path.join(test_images_dir, 'normal')

    for dir in [train_images_dir_penum, train_images_dir_norm, test_images_dir_penum, test_images_dir_norm]:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

def preprocess_image(image_array, target_size = (150, 150), 
                     normalization_value = None):

    image_array = resize(image_array, target_size, anti_aliasing=True)

    image_array = image_array/np.max(image_array)
    image_array = image_array*255

    if normalization_value is not None:

        image_array = image_array / normalization_value

    image_array = image_array.astype(np.uint8)

    return image_array


def load_csv_as_dict(csv_file):

    data_dict = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            patient_id = row[0]
            target = row[-1]
            data_dict[patient_id] = target

    return data_dict

def preprocess_dataset(root_path_to_data, scrub_pros = False):

    if scrub_pros:
        scrub_preprosessed_images(root_path_to_data)

    label_file = os.path.join(root_path_to_data, 'stage_2_train_labels.csv')
    label_dict = load_csv_as_dict(label_file)

    # Check with a single image

    # Make new directory for the preprocessed images

    all_training_images = os.listdir(os.path.join(root_path_to_data, 'stage_2_train_images'))

    # Split into training and test sets

    train_images, test_images = train_test_split(all_training_images, test_size=0.2, random_state=42)

    # Create a csv file with training and test-data from the training set

    f_name_train_test_split = 'train_test_split.csv'

    if not os.path.exists(f_name_train_test_split):

        csv_content = 'patientId,type\n'

        for i in range(len(all_training_images)):
            if all_training_images[i] in train_images:
                line_csv = all_training_images[i] + ',train' 
            else:
                line_csv = all_training_images[i] + ',test' 

            csv_content += line_csv + '\n'

        with open('train_test_split.csv', 'w') as f:
            f.write(csv_content)

    train_images_dir = os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed')
    test_images_dir = os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed')

    train_images_dir_penum = os.path.join(train_images_dir, 'penum')
    train_images_dir_norm = os.path.join(train_images_dir, 'normal')

    test_images_dir_penum = os.path.join(test_images_dir, 'penum')
    test_images_dir_norm = os.path.join(test_images_dir, 'normal')

    train_images

    img_list = load_csv_as_dict(f_name_train_test_split)

    counter = 0

    #sys.exit()

    for key in img_list.keys():

        f_name = key 
        f_name_png = key.replace('.dcm', '.png')

        img_arr = load_dicom_as_array(os.path.join(root_path_to_data, 'stage_2_train_images', f_name))
        img_arr = preprocess_image(img_arr, normalization_value=None)


        # Now sort in train, test and normal and pneumonia

        if img_list[key] == 'train':

            label_key = key.split('.')[0] # Hacky AF

            if int(label_dict[label_key]) == 0:
                imageio.imwrite(os.path.join(train_images_dir_norm, f_name_png), img_arr)
            else:
                imageio.imwrite(os.path.join(train_images_dir_penum, f_name_png), img_arr)

        else:

                if int(label_dict[label_key]) == 0:
                    imageio.imwrite(os.path.join(test_images_dir_norm, f_name_png), img_arr)
                else:
                    imageio.imwrite(os.path.join(test_images_dir_penum, f_name_png), img_arr)

        counter = counter + 1

        print("Preprocessed {} images out of {}, {}".format(
            counter, len(img_list.keys()), counter/len(img_list.keys())), end='\r')

def make_base_model(path_to_save_cnn_model = 'pneumonia_cnn.h5', save_plot = False):

    train_dataget = ImageDataGenerator(rescale=1./255)
    test_dataget = ImageDataGenerator(rescale=1./255)

    train_generator = train_dataget.flow_from_directory(
        os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_dataget.flow_from_directory(
        os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    print("Model summary:")
    model.summary()

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])

    print("Model compiled")

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50)

    model.save(path_to_save_cnn_model)

    print("Model saved to {}".format(path_to_save_cnn_model))

    if save_plot:
    # Plot the training and validation accuracy and loss

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

        plt.title('Training and validation accuracy')
        plt.legend()

        plt.savefig('training_validation_accuracy.png')

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')

        plt.title('Training and validation loss')

        plt.legend()

        plt.savefig('training_validation_loss.png')

# Make a prediction on the test set

def predict_and_plot(path_to_test_data):

    test_images_dir = os.path.join(root_path_to_data, path_to_test_data)

    test_images_dir_penum = os.path.join(test_images_dir, 'penum')
    test_images_dir_norm = os.path.join(test_images_dir, 'normal')

    test_images_penum = os.listdir(test_images_dir_penum)

    test_images_norm = os.listdir(test_images_dir_norm)

    test_images_penum = [os.path.join(test_images_dir_penum, f) for f in test_images_penum]

    test_images_norm = [os.path.join(test_images_dir_norm, f) for f in test_images_norm]

    test_images = test_images_penum + test_images_norm

    test_images_labels = [1]*len(test_images_penum) + [0]*len(test_images_norm)

    test_images_labels = np.array(test_images_labels)

    # Load the model

    model = tf.keras.models.load_model(path_to_save_cnn_model)

    # Make predictions

    predictions = model.predict(test_images)

    # Make a confusion matrix


    confusion_matrix(test_images_labels, predictions)

    # Make a ROC curve

    
    fpr, tpr, thresholds = roc_curve(test_images_labels, predictions)

    plt.plot(fpr, tpr)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.show()

# Try transfer learning from resnet

train_dataget = ImageDataGenerator(rescale=1./255)
test_dataget = ImageDataGenerator(rescale=1./255)

train_generator = train_dataget.flow_from_directory(
    os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_dataget.flow_from_directory(
    os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(150, 150, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])

model.summary()

# Assuming train_generator and validation_generator are your data generators
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

model.save('pneumonia_cnn_resnet.h5')






    








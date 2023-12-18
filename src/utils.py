import os
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import pydicom
from skimage.transform import resize
import csv
import imageio
import shutil

DIR_MAIN_IMAGES = r"rsna-pneumonia-detection-challenge"

def load_images(train=True, return_dicom_meta=False):
    import pydicom

    # dir = os.path.join(DIR_MAIN_IMAGES, "train" if train else "test")
    dir = os.path.join(DIR_MAIN_IMAGES, f"stage_2_{'''train''' if train else '''test'''}_images")
    print(dir)
    patients = os.listdir(dir)
    print("Having", len(patients), "train" if train else "test", "images / patients.")

    for pt in patients:

        print(f"LOADING PT{pt}", end="\t")
        path = os.path.join(dir, pt)

        pt = pt.split(".")[0]
        ds = pydicom.dcmread(os.path.join(path))
        img = ds.pixel_array
        print(img.shape)

        if not return_dicom_meta:
            yield pt, img
        else:
            yield pt, img, ds

    pass


def load_outcome(id_list):
    # Bounding boxes defined by x, y, width, heigh = detected pneumonia region
    #   may have multiple regions detected per patient, i.e. rows, given pneumonia present (Target=1)
    #   considering only whether pneumonia is present, for now

    y = pd.read_csv(os.path.join(DIR_MAIN_IMAGES, "stage_2_train_labels.csv"), index_col=None)
    y = y.loc[:, ["patientId", "Target"]]
    y = y.drop_duplicates()     # drop multiple entries per patient
    y = y.set_index("patientId")
    y = y["Target"]
    print("Num patient outcomes total:", y.shape, end="\t")

    # print(y["Target"].value_counts())
    # id_set_y = set(y.index.values)
    # print(len(id_set_y))
    # print(len(id_set_y.intersection(set(id_list))))

    y = y.reindex(id_list)  # retain subset, in same order as, id_list
    print("after dropping:", len(y), end="\t")
    num_per_class = np.unique(y, return_counts=True)
    print(f"per class: N_{num_per_class[0][0]}={num_per_class[1][0]} ({num_per_class[1][0] / len(y) * 100 :.1f}%) / "
          f"N_{num_per_class[0][1]}={num_per_class[1][1]} ({num_per_class[1][1] / len(y) * 100 :.1f}%)")

    return y

def pre_process_for_cnn_transfer_learning(root_path_to_data = "rsna-pneumonia-dection-challenge"):

    """
    Function to pre-process the images for transfer learning

    Parameters
    ----------

    root_path_to_data: str
        Root path to the data

    Returns
    -------
    0

    This has nested functions to not clutter the namespace, but made
    in a less than ideal way. This should be refactored in the future, but
    then it is christmas and the deadline has to be met.
    """

    def make_dirs_for_preprocessed_images(root_path_to_data):
        
        train_images_dir = os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed_hires_jpg')
        test_images_dir = os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed_hires_jpg')
    
        train_images_dir_penum = os.path.join(train_images_dir, 'penum')
        train_images_dir_norm = os.path.join(train_images_dir, 'normal')
    
        test_images_dir_penum = os.path.join(test_images_dir, 'penum')
        test_images_dir_norm = os.path.join(test_images_dir, 'normal')
    
        for dir in [train_images_dir_penum, train_images_dir_norm, test_images_dir_penum, test_images_dir_norm]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_dicom_as_array(dicom_file):
        dicom_data = pydicom.read_file(dicom_file)
        image_array = dicom_data.pixel_array
        return image_array
    
    def preprocess_image(image_array, target_size = (150, 150), 
                     normalization_value = None):

        image_array = resize(image_array, target_size, anti_aliasing=True)

        image_array = image_array/np.max(image_array)
        image_array = image_array*255

        if normalization_value is not None:

            image_array = image_array / normalization_value

        image_array = image_array.astype(np.uint8)

        return image_array
    
    def scrub_preprosessed_images(root_path_to_data):
        """
        Delete the content of the directories where the preprocessed images are stored.
        """

        train_images_dir = os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed_hires_jpg')
        test_images_dir = os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed_hires_jpg')

        train_images_dir_penum = os.path.join(train_images_dir, 'penum')
        train_images_dir_norm = os.path.join(train_images_dir, 'normal')

        test_images_dir_penum = os.path.join(test_images_dir, 'penum')
        test_images_dir_norm = os.path.join(test_images_dir, 'normal')

        for dir in [train_images_dir_penum, train_images_dir_norm, test_images_dir_penum, test_images_dir_norm]:
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))

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
    
    def preprocess_dataset(root_path_to_data, scrub_pros = False, 
                           use_pre_defined_split = True):

        if scrub_pros:
            scrub_preprosessed_images(root_path_to_data)

        label_file = os.path.join(root_path_to_data, 'stage_2_train_labels.csv')
        label_dict = load_csv_as_dict(label_file)

        # Make new directory for the preprocessed images

        all_training_images = os.listdir(os.path.join(root_path_to_data, 'stage_2_train_images'))

        if use_pre_defined_split:
            f_name_train_test_split = "train_test_split_segmented.csv"
            img_list = load_csv_as_dict(f_name_train_test_split)

            train_images_dir = os.path.join(root_path_to_data, 'stage_2_train_images_preprocessed_hires_jpg')
            test_images_dir = os.path.join(root_path_to_data, 'stage_2_test_images_preprocessed_hires_jpg')
    
            train_images_dir_penum = os.path.join(train_images_dir, 'penum')
            train_images_dir_norm = os.path.join(train_images_dir, 'normal')
    
            test_images_dir_penum = os.path.join(test_images_dir, 'penum')
            test_images_dir_norm = os.path.join(test_images_dir, 'normal')

        else:

            print("Error, not yet implemented")

        img_list = load_csv_as_dict(f_name_train_test_split)

        counter = 0

        for key in img_list.keys():

            f_name = key + ".dcm"
            f_name_png = f_name.replace('.dcm', '.jpg')

            img_arr = load_dicom_as_array(os.path.join(root_path_to_data, 'stage_2_train_images', f_name))
            img_arr = img_arr/255.0

            # Now sort in train, test and normal and pneumonia

            if img_list[key] == 'train':

                label_key = key.split('.')[0] # Hacky AF

                if int(label_dict[label_key]) == 0:
                    imageio.imwrite(os.path.join(train_images_dir_norm, f_name_png), img_arr)
                else:
                    imageio.imwrite(os.path.join(train_images_dir_penum, f_name_png), img_arr)

            else:
                    label_key = key.split('.')[0] # Hacky AF

                    if int(label_dict[label_key]) == 0:
                        imageio.imwrite(os.path.join(test_images_dir_norm, f_name_png), img_arr)
                    else:
                        imageio.imwrite(os.path.join(test_images_dir_penum, f_name_png), img_arr)

            counter = counter + 1

            print("Preprocessed {} images out of {}, {}".format(
                counter, len(img_list.keys()), counter/len(img_list.keys())), end='\r')
            
    make_dirs_for_preprocessed_images(root_path_to_data)
    preprocess_dataset(root_path_to_data, scrub_pros = True, 
                           use_pre_defined_split = True)
    


    


    return 0
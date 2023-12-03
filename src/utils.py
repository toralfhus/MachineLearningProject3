import os
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt


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

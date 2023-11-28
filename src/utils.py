import os
import numpy as np
import sys


DIR_MAIN_IMAGES = r"C:\rsna-miccai-brain-tumor-radiogenomic-classification"


def load_images(train=True, image_mode="FLAIR", return_dicom=False):
    import pydicom

    dir = os.path.join(DIR_MAIN_IMAGES, "train" if train else "test")
    print(dir)
    patients = os.listdir(dir)
    print("Having", len(patients), "train" if train else "test", "patients:", patients)


    for pt in patients:

        print(f"LOADING PT{pt} {image_mode}", end="\t")
        path = os.path.join(dir, pt, image_mode)

        # Sort images by name (verify dicom positioning?)
        images = [int(nm.split("-")[-1].split(".")[0]) for nm in os.listdir(path)]
        images.sort()
        images = [f"Image-{i}.dcm" for i in images]


        voxels = []

        for im in images:
            ds = pydicom.dcmread(os.path.join(path, im))
            arr = ds.pixel_array
            voxels.append(arr)
        voxels = np.array(voxels)
        print(voxels.shape)

        if not return_dicom:
            yield pt, voxels
        else:
            yield pt, voxels, ds

    pass


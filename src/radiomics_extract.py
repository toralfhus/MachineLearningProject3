import numpy as np
from radiomics import featureextractor    # pyradiomics

from matplotlib import pyplot as plt
import pandas as pd
from utils import *
import SimpleITK as sitk
from time import time
from segmentation import segment_unet


# NAME = f"radiomics_fbc=32"
# NAME = f"radiomics_downsamp_fbc=32"
NAME = f"radiomics_seg_fbc=32"
DIR_IMAGES = os.path.join(DIR_MAIN_IMAGES, "stage_2_train_images_preprocessed")

shapes = []
spacings = []
thicknesses = []
i = 0

for traintest in ["train"]:
    do_train = True if traintest == "train" else False
    SAVENAME = NAME + "_" + traintest + ".csv"
    SAVENAME_BU = NAME + "_" + traintest + "_bu.csv"   # backup

    try:
        df_radiomics = pd.read_csv(SAVENAME, index_col=0)
    except Exception:
        df_radiomics = pd.DataFrame()
    # print(df_radiomics)
    print("LOADED:", df_radiomics.shape)
    for pt, img in load_images(dir=os.path.join(DIR_IMAGES), return_dicom_meta=False):
        i += 1
        print(i, pt, end="\t")
        print(pt in df_radiomics.index)

        if not pt in df_radiomics.index:
            try:
                t0 = time()
                # print(np.round(ds.PixelSpacing, 2))
                # print(round(ds.SliceThickness, 2))

                # REMOVE ZERO-VALUED BACKGROUND (VERY SIMPLE "SEGMENTATION")
                # msk = np.zeros(img.shape)
                # msk[img != 0] = 1
                # print(np.count_nonzero(msk))

                # INCLUDE ALL PIXELS
                # msk = np.ones(img.shape)
                # msk[0, 0] = 0   # need to have at least one pixel not in mask to work :))
                msk = segment_unet(img)


                # plt.imshow(msk[msk.shape[0] // 2], alpha=0.5)
                # plt.imshow(img[img.shape[0] // 2], alpha=1)
                # plt.show()

                extractor = featureextractor.RadiomicsFeatureExtractor("radiomics_settings.yaml")
                print(extractor.settings)

                imgshape = img.shape
                img, msk = sitk.GetImageFromArray(img), sitk.GetImageFromArray(msk)
                features = extractor.execute(img, msk)
                print(features)

                fts_to_save = features.keys()
                fts_to_save = list(filter(lambda ft: "diagnostics" not in ft, fts_to_save))
                print(fts_to_save)

                for ft in fts_to_save:
                    val = features[ft]
                    df_radiomics.loc[pt, ft] = val


                df_radiomics.loc[pt, "shape"] = str(imgshape)
                # df_radiomics.loc[pt, "PixelSpacing"] = str(np.round(ds.PixelSpacing, 3))
                # df_radiomics.loc[pt, "SliceThickness"] = str(ds.SliceThickness)
                print(df_radiomics)

                # Saving dataframe + backup, on separate iterations to avoid data loss if system breakdown
                if not((i+1)%1000):
                    df_radiomics.to_csv(SAVENAME_BU)
                elif not(i%250):
                    df_radiomics.to_csv(SAVENAME)

                t1 = time()
                dt = t1 - t0
                print(f"SAVED. Time extraction image = {dt:.1f} s, whole extraction=")
            except Exception as e:
                print(*e.args)
        del pt, img
df_radiomics.to_csv(SAVENAME)

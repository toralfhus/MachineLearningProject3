import numpy as np
from radiomics import featureextractor    # pyradiomics

from matplotlib import pyplot as plt
import pandas as pd
from utils import *
import SimpleITK as sitk


shapes = []
spacings = []
thicknesses = []
i = 0

for traintest in ["train", "test"]:
    df_radiomics = pd.DataFrame()
    do_train = True if traintest == "train" else False
    for pt, img, ds in load_images(train=do_train, return_dicom_meta=True):
        i += 1
        print(i, pt, end="\t")

        try:

            print(np.round(ds.PixelSpacing, 2))
            # print(round(ds.SliceThickness, 2))

            # REMOVE ZERO-VALUED BACKGROUND (VERY SIMPLE "SEGMENTATION")
            # msk = np.zeros(img.shape)
            # msk[img != 0] = 1
            # print(np.count_nonzero(msk))

            # INCLUDE ALL PIXELS
            msk = np.ones(img.shape)
            msk[0, 0] = 0   # need to have at least one pixel not in mask to work :))

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
            df_radiomics.loc[pt, "PixelSpacing"] = str(np.round(ds.PixelSpacing, 3))
            # df_radiomics.loc[pt, "SliceThickness"] = str(ds.SliceThickness)
            print(df_radiomics)

            df_radiomics.to_csv(f"radiomics_fbc=32_{traintest}.csv")
        except Exception as e:
            print(*e.args)

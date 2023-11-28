from radiomics import featureextractor    # pyradiomics

from matplotlib import pyplot as plt
import pandas as pd
from utils import *
import SimpleITK as sitk


shapes = []
spacings = []
thicknesses = []
i = 0
df_meta = pd.DataFrame(dtype="object")

for traintest in ["train", "test"]:
    df_radiomics = pd.DataFrame()
    do_train = True if traintest == "train" else False
    for pt, img, ds in load_images(train=do_train, image_mode="T1w", return_dicom=True):
        i += 1
        print(i, pt, end="\t")
        print(np.round(ds.PixelSpacing, 2), round(ds.SliceThickness, 2))
        df_meta.loc[i, "shape"] = str(img.shape)
        df_meta.loc[i, "PixelSpacing"] = str(np.round(ds.PixelSpacing, 3))
        df_meta.loc[i, "SliceThickness"] = str(ds.SliceThickness)


        # REMOVE ZERO-VALUED BACKGROUND (VERY SIMPLE "SEGMENTATION")
        msk = np.zeros(img.shape)
        msk[img != 0] = 1

        # plt.imshow(msk[msk.shape[0] // 2], alpha=0.5)
        # plt.imshow(img[img.shape[0] // 2], alpha=0.5)
        # plt.show()

        extractor = featureextractor.RadiomicsFeatureExtractor("radiomics_settings.yaml")
        print(extractor.settings)

        img, msk = sitk.GetImageFromArray(img), sitk.GetImageFromArray(msk)
        features = extractor.execute(img, msk)
        print(features)

        fts_to_save = features.keys()
        fts_to_save = list(filter(lambda ft: "diagnostics" not in ft, fts_to_save))
        print(fts_to_save)

        for ft in fts_to_save:
            val = features[ft]
            df_radiomics.loc[pt, ft] = val


        # if i > 10:
        #     break
        print(df_radiomics)

    print(df_meta)
    print(df_radiomics)

    df_radiomics.to_csv(f"radiomics_fbc=32_{traintest}.csv")
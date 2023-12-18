from utils import *
from matplotlib import pyplot as plt
import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import active_contour
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler


nn_seg_trained = "segmentator.hdf5"  # https://github.com/imlab-uiip/lung-segmentation-2d
global nn_seg
nn_seg = load_model(nn_seg_trained)


def segment_unet(img, thresh=0.9):
    # Using a pre-trained U-net inspired CNN to segment the lungs in 2D radiographic images
    # https://github.com/imlab-uiip/lung-segmentation-2d
    shape = img.shape
    img = StandardScaler().fit_transform(img.reshape(-1, 1)).reshape(shape)
    X = img.reshape(1, *shape, 1)
    seg = nn_seg.predict(X)
    seg = seg.reshape(shape)
    seg[seg >= thresh] = 1
    seg[seg < thresh] = 0
    print("IMAGE SEGMENTED using pre-trained 2D U-net")
    return seg



if __name__ == "__main__":

    gen = load_images(os.path.join(DIR_MAIN_IMAGES, "stage_2_train_images_preprocessed"))


    for pt, img in gen:
        print(pt, img.shape, end="\t")
        shape = img.shape


        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        ax[0].imshow(img)
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)  # blur image using Gaussian 3x3 filter
        ax[1].imshow(img_blur)
        thr, seg = cv2.threshold(img_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"otsu={thr:.0f}")
        ax[2].imshow(seg)
        fig.suptitle("Otsu")
        print(img.shape)


        # img = StandardScaler().fit_transform(img.reshape(-1, 1)).reshape(shape)

        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        ax[0].imshow(img)

        seg = segment_unet(img)
        # gen_keras = ImageDataGenerator()
        # X = gen_keras.flow(X, batch_size=1)
        ax[1].imshow(seg)

        [axx.axis("off") for axx in ax]
        fig.tight_layout()

        # fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        # ax[0].imshow(img)

        # edges = cv2.Canny(img, 30, 150)
        # print(np.shape(edges))
        # ax[1].imshow(edges)
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print([np.shape(c) for c in contours])
        #
        #
        # snake_initial = np.vstack(contours)
        # # Reshape the snake to fit the active_contour input format
        # snake_initial = np.squeeze(snake_initial)
        # init = np.array([snake_initial[:, 1], snake_initial[:, 0]]).T
        #
        # # Apply active contour model (snake) to find lung boundaries
        # snake = active_contour(img, init, alpha=0.1, beta=1.0, gamma=0.01)
        # print(np.shape(snake))
        #
        #
        # # Create a mask from the snake
        # mask = np.zeros_like(img, dtype=np.uint8)
        # snake_int = np.round(snake).astype(np.int32)
        # cv2.drawContours(mask, [snake_int], 0, 255, -1)
        #
        # # Apply the mask to the original image
        # segmented_lungs = cv2.bitwise_and(img, mask)
        # print(segmented_lungs.shape)
        # ax[2].imshow(segmented_lungs)

        plt.show()
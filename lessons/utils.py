import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
import ntpath


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 3:
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

    # Define a function to return HOG features and visualization


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features


        # images = glob.glob('images_data/*.jpeg')
        # cars = []
        # notcars = []
        # for image in images:
        #     head, tail = ntpath.split(image)
        #     if 'image' in tail or 'extra' in tail:
        #         notcars.append(image)
        #     else:
        #         cars.append(image)
        #
        # car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
        #                                 hist_bins=32, hist_range=(0, 256))
        # notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
        #                                    hist_bins=32, hist_range=(0, 256))
        #
        # if len(car_features) > 0:
        #     # Create an array stack of feature vectors
        #     X = np.vstack((car_features, notcar_features)).astype(np.float64)
        #     # Fit a per-column scaler
        #     X_scaler = StandardScaler().fit(X)
        #     # Apply the scaler to X
        #     scaled_X = X_scaler.transform(X)
        #     car_ind = np.random.randint(0, len(cars))
        #     # Plot an example of raw and scaled features
        #     fig = plt.figure(figsize=(12, 4))
        #     plt.subplot(131)
        #     plt.imshow(mpimg.imread(cars[car_ind]))
        #     plt.title('Original Image')
        #     plt.subplot(132)
        #     plt.plot(X[car_ind])
        #     plt.title('Raw Features')
        #     plt.subplot(133)
        #     plt.plot(scaled_X[car_ind])
        #     plt.title('Normalized Features')
        #     fig.tight_layout()
        #     plt.show()
        # else:
        #     print('Your function only returns empty feature vectors...')

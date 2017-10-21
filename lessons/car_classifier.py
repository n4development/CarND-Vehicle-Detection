import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import ntpath
import utils
import traning_model
from sklearn.preprocessing import StandardScaler

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

images = glob.glob('images_data/*.jpeg')
print(len(images))
cars = []
notcars = []

for image in images:
    head, tail = ntpath.split(image)
    if 'image' in tail or 'extra' in tail:
        notcars.append(image)
    else:
        cars.append(image)

print('* images analysis ',
      len(cars), ' cars and',
      len(notcars), ' non-cars')


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


data_info = data_look(cars, notcars)

print('* function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('* image size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])

sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
colorspace = 'HSV'
# Define HOG parameters
orient = 10
pix_per_cell = 8
cell_per_block = 2
# Can be 0, 1, 2, 3
hog_channel = 3
spatial = 32
histbin = 32

# car_ind = np.random.randint(0, len(cars))
# notcar_ind = np.random.randint(0, len(notcars))
car_features = utils.extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
notcar_features = utils.extract_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)


# Create an array stack of feature vectors
feature_list = [car_features, notcar_features]
X = np.vstack(feature_list).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
car_ind = np.random.randint(0, len(cars))
# Plot an example of raw and scaled features
# fig = plt.figure(figsize=(12,4))
# plt.subplot(131)
# plt.imshow(mpimg.imread(cars[car_ind]))
# plt.title('Original Image')
# plt.subplot(132)
# plt.plot(X[car_ind])
# plt.title('Raw Features')
# plt.subplot(133)
# plt.plot(scaled_X[car_ind])
# plt.title('Normalized Features')
# fig.tight_layout()

# # Read in car / not-car images
# car_image = mpimg.imread(cars[car_ind])
# notcar_image = mpimg.imread(notcars[notcar_ind])
#
# gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
# # Call our function with vis=True to see an image output
# features, hog_image = utils.get_hog_features(gray, orient,
#                                              pix_per_cell, cell_per_block,
#                                              vis=True, feature_vec=False)
print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# traning_model.color_classifier(X=scaled_X, Y=y, spatial=spatial, histbin=histbin)
traning_model.HOG_classifier(X=scaled_X, Y=y)

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(car_image)
# plt.title('Example Car Image')
# plt.subplot(122)
# plt.imshow(notcar_image)
# plt.title('Example Not-car Image')


# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(gray, cmap='gray')
# plt.title('Example Car Image')
# plt.subplot(122)
# plt.imshow(hog_image, cmap='gray')
# plt.title('HOG Visualization')
#
# plt.show()

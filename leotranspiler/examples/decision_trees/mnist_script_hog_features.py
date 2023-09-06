# -*- coding: utf-8 -*-

import gzip
import os
import shutil

import requests


def download_and_extract_dataset(url, save_path, folder_path):
    """Download and extract dataset if it doesn't exist."""
    if not os.path.exists(save_path):
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url)
        with open(save_path, "wb") as file:
            file.write(response.content)

        decompressed_file_name = os.path.splitext(os.path.basename(save_path))[0]
        decompressed_file_path = os.path.join(folder_path, decompressed_file_name)

        with gzip.open(save_path, "rb") as f_in:
            with open(decompressed_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"{decompressed_file_name} downloaded and extracted.")
    else:
        print(f"{os.path.basename(save_path)} already exists.")


file_info = [
    (
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ),
    (
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    (
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
    (
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ),
]

folder_name = "tmp/mnist"
folder_path = os.path.join(os.getcwd(), folder_name)

os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

# Download and extract each file
for url, file_name in file_info:
    path_to_save = os.path.join(folder_path, file_name)
    download_and_extract_dataset(url, path_to_save, folder_path)

# %%
import numpy as np


def read_idx3_ubyte_image_file(filename):
    """Read IDX3-ubyte formatted image data."""
    with open(filename, "rb") as f:
        magic_num = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        num_cols = int.from_bytes(f.read(4), byteorder="big")

        if magic_num != 2051:
            raise ValueError(f"Invalid magic number: {magic_num}")

        images = np.zeros((num_images, num_rows, num_cols), dtype=np.uint8)

        for i in range(num_images):
            for r in range(num_rows):
                for c in range(num_cols):
                    pixel = int.from_bytes(f.read(1), byteorder="big")
                    images[i, r, c] = pixel

    return images


def read_idx1_ubyte_label_file(filename):
    """Read IDX1-ubyte formatted label data."""
    with open(filename, "rb") as f:
        magic_num = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        if magic_num != 2049:
            raise ValueError(f"Invalid magic number: {magic_num}")

        labels = np.zeros(num_labels, dtype=np.uint8)

        for i in range(num_labels):
            labels[i] = int.from_bytes(f.read(1), byteorder="big")

    return labels


# Example usage
folder_path = os.path.join(
    os.getcwd(), folder_name
)  # Adjust this path to where you stored the files

train_images = read_idx3_ubyte_image_file(
    os.path.join(folder_path, "train-images-idx3-ubyte")
)
train_labels = read_idx1_ubyte_label_file(
    os.path.join(folder_path, "train-labels-idx1-ubyte")
)
test_images = read_idx3_ubyte_image_file(
    os.path.join(folder_path, "t10k-images-idx3-ubyte")
)
test_labels = read_idx1_ubyte_label_file(
    os.path.join(folder_path, "t10k-labels-idx1-ubyte")
)

print(
    f"Shape of train_images: {train_images.shape}"
)  # Should output "Shape of train_images: (60000, 28, 28)"
print(
    f"Shape of train_labels: {train_labels.shape}"
)  # Should output "Shape of train_labels: (60000,)"
print(
    f"Shape of test_images: {test_images.shape}"
)  # Should output "Shape of test_images: (10000, 28, 28)"
print(
    f"Shape of test_labels: {test_labels.shape}"
)  # Should output "Shape of test_labels: (10000,)"

# %%
# Reshape the datasets from 3D to 2D
train_images_2d = train_images.reshape(
    train_images.shape[0], -1
)  # -1 infers the size from the remaining dimensions
test_images_2d = test_images.reshape(test_images.shape[0], -1)

# %%
from scipy.ndimage import label

# Create the classifier and fit it to the reshaped training data
from sklearn.tree import DecisionTreeClassifier


def average_intensity(image):
    return np.mean(image)


def num_white_regions(image, threshold=128):
    # Threshold the image to convert it to binary
    bin_image = (image > threshold).astype(int)
    # Use scipy's label function to count connected regions
    labeled_array, num_features = label(bin_image)
    return num_features


from scipy.stats import kurtosis, skew


def standard_deviation(image):
    return np.std(image)


def skewness(image):
    return skew(image)


def kurtosis_value(image):
    return kurtosis(image)


def aspect_ratio(image, threshold=128):
    # Threshold the image to create a binary representation
    bin_image = image > threshold
    # Find the bounding box
    row_indices, col_indices = np.nonzero(bin_image)
    max_row, min_row = np.max(row_indices), np.min(row_indices)
    max_col, min_col = np.max(col_indices), np.min(col_indices)

    # Calculate the aspect ratio of the bounding box
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    if height == 0:  # To avoid division by zero
        return 1.0

    return width / height


from scipy.ndimage import label
from scipy.optimize import curve_fit


def linear_func(x, a, b):
    return a * x + b


def avg_dist_to_fitted_line(image, threshold=128):
    # Threshold the image
    bin_image = (image > threshold).astype(int)

    # Identify connected components
    labeled_array, num_features = label(bin_image)

    # Find the largest connected component
    largest_component = None
    max_count = 0
    for i in range(1, num_features + 1):
        component = np.where(labeled_array == i)
        count = len(component[0])
        if count > max_count:
            max_count = count
            largest_component = component

    if largest_component is None:
        return 0.0

    x, y = largest_component
    if len(x) <= 1:  # Can't fit a line to a single point or empty set
        return 0.0

    # Fit a line to the largest component
    popt, _ = curve_fit(linear_func, x, y)

    # Calculate the average distance from each point to the line
    distances = np.abs(y - linear_func(x, *popt))
    avg_distance = np.mean(distances)

    return avg_distance


from skimage.feature import hog


def compute_hog_features(image):
    # The image shape is 28x28
    # You can experiment with these parameters
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    nbins = 6  # Number of orientation bins

    features, hog_image = hog(
        image,
        orientations=nbins,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
    )
    return features


# Compute HoG features for the training set
hog_features_train = np.array([compute_hog_features(img) for img in train_images])

# Compute HoG features for the test set
hog_features_test = np.array([compute_hog_features(img) for img in test_images])


# Calculate new features for training set
avg_intensity_train = np.array([average_intensity(img) for img in train_images_2d])
white_regions_train = np.array([num_white_regions(img) for img in train_images_2d])

# Calculate new features for test set
avg_intensity_test = np.array([average_intensity(img) for img in test_images_2d])
white_regions_test = np.array([num_white_regions(img) for img in test_images_2d])

# Calculate new features for training set
std_dev_train = np.array([standard_deviation(img) for img in train_images_2d])
skewness_train = np.array([skewness(img) for img in train_images_2d])
kurtosis_train = np.array([kurtosis_value(img) for img in train_images_2d])

# Calculate new features for test set
std_dev_test = np.array([standard_deviation(img) for img in test_images_2d])
skewness_test = np.array([skewness(img) for img in test_images_2d])
kurtosis_test = np.array([kurtosis_value(img) for img in test_images_2d])

# Calculate new features for training set
aspect_ratio_train = np.array([aspect_ratio(img) for img in train_images])

# Calculate new features for test set
aspect_ratio_test = np.array([aspect_ratio(img) for img in test_images])

# Calculate new features for training set
avg_dist_train = np.array([avg_dist_to_fitted_line(img) for img in train_images])

# Calculate new features for test set
avg_dist_test = np.array([avg_dist_to_fitted_line(img) for img in test_images])


# Combine old and new features for training set
# X_new_train = np.c_[avg_intensity_train, white_regions_train, aspect_ratio_train, hog_features_train]

# Combine old and new features for test set
# X_new_test = np.c_[avg_intensity_test, white_regions_test, aspect_ratio_test, hog_features_test]

X_new_train = np.c_[hog_features_train]
X_new_test = np.c_[hog_features_test]


# Train a Decision Tree classifier
tree_clf = DecisionTreeClassifier(max_depth=10)
tree_clf.fit(X_new_train, train_labels)

# Evaluate the classifier
print("Train score:", tree_clf.score(X_new_train, train_labels))
print("Test score:", tree_clf.score(X_new_test, test_labels))

from leotranspiler import LeoTranspiler

# %%
lt = LeoTranspiler(model=tree_clf, validation_data=train_images_2d[0:50])
leo_project_path = os.path.join(os.getcwd(), "tmp/mnist")
leo_project_name = "tree_credit"
lt.to_leo(path=leo_project_path, project_name=leo_project_name)

# %%
# prove and compare the Leo prediction with the Python prediction and the label
zkp = lt.execute(input_sample=X_new_train[0])
python_prediction = tree_clf.predict([X_new_train[0]])

print(f"Circuit constraints: {zkp.circuit_constraints}")
print(f"Active input count: {zkp.active_input_count}")
print(f"Leo prediction in fixed-point notation: {zkp.output[0]}")
print(f"Leo prediction in decimal notation: {zkp.output_decimal[0]}")
print(f"Python prediction: {python_prediction[0]}")
print(f"Label: {test_labels[0]}")

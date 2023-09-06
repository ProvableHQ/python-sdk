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

import cv2
import numpy as np

# %%
from scipy.ndimage import label

# Create the classifier and fit it to the reshaped training data
from sklearn.tree import DecisionTreeClassifier


def extract_haar_features(image, step_size=8, feature_type="both"):
    # Compute the integral image
    int_img = cv2.integral(image)

    # Define a small set of Haar-like features
    features = []

    def horizontal_features(y, x, w, h):
        A = int_img[y, x]
        B = int_img[y, x + w // 2]
        C = int_img[y + h, x]
        D = int_img[y + h, x + w // 2]
        E = int_img[y, x + w]
        F = int_img[y + h, x + w]
        return (D - B + A) - (F - D + E - B)

    def vertical_features(y, x, w, h):
        A = int_img[y, x]
        B = int_img[y, x + w]
        C = int_img[y + h // 2, x]
        D = int_img[y + h // 2, x + w]
        E = int_img[y + h, x]
        F = int_img[y + h, x + w]
        return (D - B + A) - (F - D + E - C)

    for y in range(0, 28, step_size):
        for x in range(0, 28, step_size):
            for w in range(step_size, 28 - x, step_size):
                for h in range(step_size, 28 - y, step_size):
                    if feature_type in ["horizontal", "both"]:
                        features.append(horizontal_features(y, x, w, h))
                    if feature_type in ["vertical", "both"]:
                        features.append(vertical_features(y, x, w, h))

    return np.array(features)


# Test the function on a single 28x28 image
single_image = train_images[0]
haar_features = extract_haar_features(single_image, step_size=8, feature_type="both")
print(
    "Reduced Haar Features Length for Both Horizontal and Vertical:", len(haar_features)
)


# Compute HoG features for the training set
hog_features_train = np.array([extract_haar_features(img) for img in train_images])

# Compute HoG features for the test set
hog_features_test = np.array([extract_haar_features(img) for img in test_images])


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

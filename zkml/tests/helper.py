# -*- coding: utf-8 -*-
# noqa: D100

import copy
library_name = "zkml"


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np

# Define the PyTorch neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prune_neural_network(
    network_to_prune, weight_threshold=1e-1, bias_threshold=1e-1
):  # noqa: D103
    num_weights = 0
    num_changed_weights = 0
    num_biases = 0
    num_changed_biases = 0

    # Pruning the weights
    for i in range(len(network_to_prune.coefs_)):
        flattened_weights = network_to_prune.coefs_[i].ravel()
        for j, weight in enumerate(flattened_weights):
            if abs(weight) < weight_threshold:
                flattened_weights[j] = 0
                num_changed_weights += 1
            num_weights += 1

        # Reshape back to the original shape
        network_to_prune.coefs_[i] = flattened_weights.reshape(
            network_to_prune.coefs_[i].shape
        )

    # Pruning the biases
    for i in range(len(network_to_prune.intercepts_)):
        flattened_biases = network_to_prune.intercepts_[i].ravel()
        for j, bias in enumerate(flattened_biases):
            if abs(bias) < bias_threshold:
                flattened_biases[j] = 0
                num_changed_biases += 1
            num_biases += 1

        # Reshape back to the original shape (this might not be needed for biases,
        # but I'm including it for clarity)
        network_to_prune.intercepts_[i] = flattened_biases.reshape(
            network_to_prune.intercepts_[i].shape
        )

    print(f"Number of weight parameters: {num_weights}")  # noqa: T201
    print(f"Number of changed weight parameters: {num_changed_weights}")  # noqa: T201
    print(f"Number of bias parameters: {num_biases}")  # noqa: T201
    print(f"Number of changed bias parameters: {num_changed_biases}")  # noqa: T201
    print(  # noqa: T201
        "Percentage of weights pruned: {:.2f}%".format(
            num_changed_weights / num_weights * 100
        )
    )
    print(  # noqa: T201
        "Percentage of biases pruned: {:.2f}%".format(
            num_changed_biases / num_biases * 100
        )
    )
    print(  # noqa: T201
        "Remaining number of non-zero weights: {}".format(
            num_weights - num_changed_weights
        )
    )
    print(  # noqa: T201
        "Remaining number of non-zero biases: {}".format(
            num_biases - num_changed_biases
        )
    )
    return network_to_prune


def prune_pytorch_network(
    model, weight_threshold=1e-1, bias_threshold=1e-1
):  # noqa: D103
    num_weights = 0
    num_changed_weights = 0
    num_biases = 0
    num_changed_biases = 0

    # Pruning the weights
    for name, param in model.named_parameters():
        if "weight" in name:
            flattened_weights = param.data.view(-1)
            for j, weight in enumerate(flattened_weights):
                if abs(weight) < weight_threshold:
                    flattened_weights[j] = 0
                    num_changed_weights += 1
                num_weights += 1
            param.data = flattened_weights.view(param.data.shape)
        elif "bias" in name:
            flattened_biases = param.data.view(-1)
            for j, bias in enumerate(flattened_biases):
                if abs(bias) < bias_threshold:
                    flattened_biases[j] = 0
                    num_changed_biases += 1
                num_biases += 1
            param.data = flattened_biases.view(param.data.shape)

    print(f"Number of weight parameters: {num_weights}")  # noqa: T201
    print(f"Number of changed weight parameters: {num_changed_weights}")  # noqa: T201
    print(f"Number of bias parameters: {num_biases}")  # noqa: T201
    print(f"Number of changed bias parameters: {num_changed_biases}")  # noqa: T201
    print(  # noqa: T201
        f"Percentage of weights pruned: {num_changed_weights / num_weights * 100:.2f}%"
    )
    print(  # noqa: T201
        f"Percentage of biases pruned: {num_changed_biases / num_biases * 100:.2f}%"
    )
    print(  # noqa: T201
        f"Remaining number of non-zero weights: {num_weights - num_changed_weights}"
    )
    print(  # noqa: T201
        f"Remaining number of non-zero biases: {num_biases - num_changed_biases}"
    )

    return model

def prepare_MNIST_haar():
    import gzip
    import os
    import shutil

    import requests

    def download_and_extract_dataset(url, save_path, folder_path):
        """Download and extract dataset if it doesn't exist."""
        if not os.path.exists(save_path):
            print(f"Downloading {os.path.basename(save_path)}...")  # noqa: T201
            response = requests.get(url)
            with open(save_path, "wb") as file:
                file.write(response.content)

            decompressed_file_name = os.path.splitext(os.path.basename(save_path))[
                0
            ]
            decompressed_file_path = os.path.join(
                folder_path, decompressed_file_name
            )

            with gzip.open(save_path, "rb") as f_in:
                with open(decompressed_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print(  # noqa: T201
                f"{decompressed_file_name} downloaded and extracted."
            )  # noqa: T201
        else:
            print(f"{os.path.basename(save_path)} already exists.")  # noqa: T201

    # URLs and filenames
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

    folder_name = library_name + "/tests/tmp/mnist"
    folder_path = os.path.join(os.getcwd(), folder_name)

    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

    # Download and extract each file
    for url, file_name in file_info:
        path_to_save = os.path.join(folder_path, file_name)
        download_and_extract_dataset(url, path_to_save, folder_path)


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

    print(  # noqa: T201
        f"Shape of train_images: {train_images.shape}"
    )  # Should output "Shape of train_images: (60000, 28, 28)"
    print(  # noqa: T201
        f"Shape of train_labels: {train_labels.shape}"
    )  # Should output "Shape of train_labels: (60000,)"
    print(  # noqa: T201
        f"Shape of test_images: {test_images.shape}"
    )  # Should output "Shape of test_images: (10000, 28, 28)"
    print(  # noqa: T201
        f"Shape of test_labels: {test_labels.shape}"
    )  # Should output "Shape of test_labels: (10000,)"

    # Reshape the datasets from 3D to 2D
    train_images_2d = train_images.reshape(
        train_images.shape[0], -1
    )  # -1 infers the size from the remaining dimensions
    test_images_2d = test_images.reshape(test_images.shape[0], -1)



    # conver to pytorch tensors
    import torch

    train_images_tensor_initial = torch.from_numpy(train_images_2d).float()
    train_labels_tensor_initial = torch.from_numpy(train_labels).long()
    test_images_tensor = torch.from_numpy(test_images_2d).float()
    test_labels_tensor = torch.from_numpy(test_labels).long()

    # seed the random number generator
    torch.manual_seed(0)

    # shuffle the training dataset
    indices = torch.randperm(train_images_tensor_initial.shape[0])
    train_images_tensor_shuffled = train_images_tensor_initial[indices]
    train_labels_tensor_shuffled = train_labels_tensor_initial[indices]

    # get a 10% validation set
    validation_size = int(train_images_tensor_shuffled.shape[0] * 0.1)
    validation_images_tensor = train_images_tensor_shuffled[:validation_size]
    validation_labels_tensor = train_labels_tensor_shuffled[:validation_size]
    train_images_tensor = train_images_tensor_shuffled[validation_size:]
    train_labels_tensor = train_labels_tensor_shuffled[validation_size:]

    def get_bounding_box(img):
        """
        Extract the bounding box from an MNIST image.

        Args:
            img (np.ndarray): 2D numpy array representing the MNIST image.

        Returns:
            (np.ndarray): Cropped image with the bounding box.
        """

        # convert torch image to numpy array
        img = img.numpy()

        # Find the rows and columns where the image has non-zero pixels
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)

        # Find the first and last row and column indices where the image has non-zero pixels
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Return the cropped image
        return img[rmin : rmax + 1, cmin : cmax + 1]
    
    image_id = 0
    image = train_images_tensor[image_id].reshape(28, 28)
    cropped_image = get_bounding_box(image)

    import cv2

    cropped_image_uint8 = np.clip(cropped_image, 0, 255).astype(np.uint8)
    resized_image = cv2.resize(cropped_image_uint8, (20, 20), interpolation=cv2.INTER_AREA)

    num_train = len(train_images_tensor)

    train_images_tensor_resized = np.zeros((num_train, 400))

    for i in range(num_train):
        cropped_image = get_bounding_box(train_images_tensor[i].reshape(28, 28))
        cropped_image_uint8 = np.clip(cropped_image, 0, 255).astype(np.uint8)
        resized_image = cv2.resize(
            cropped_image_uint8, (20, 20), interpolation=cv2.INTER_AREA
        )
        train_images_tensor_resized[i, :] = resized_image.flatten()

    num_test = len(test_images_tensor)

    num_val = len(validation_images_tensor)

    validation_images_tensor_resized = np.zeros((num_val, 400))

    for i in range(num_val):
        cropped_image = get_bounding_box(validation_images_tensor[i].reshape(28, 28))
        cropped_image_uint8 = np.clip(cropped_image, 0, 255).astype(np.uint8)
        resized_image = cv2.resize(
            cropped_image_uint8, (20, 20), interpolation=cv2.INTER_AREA
        )
        validation_images_tensor_resized[i, :] = resized_image.flatten()

    num_test = len(test_images_tensor)

    test_images_tensor_resized = np.zeros((num_test, 400))

    for i in range(num_test):
        cropped_image = get_bounding_box(test_images_tensor[i].reshape(28, 28))
        cropped_image_uint8 = np.clip(cropped_image, 0, 255).astype(np.uint8)
        resized_image = cv2.resize(
            cropped_image_uint8, (20, 20), interpolation=cv2.INTER_AREA
        )
        test_images_tensor_resized[i, :] = resized_image.flatten()

    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_features_normalized = torch.tensor(scaler.fit_transform(train_images_tensor))
    val_features_normalized = torch.tensor(scaler.transform(validation_images_tensor))
    test_features_normalized = torch.tensor(scaler.transform(test_images_tensor))

    train_features_resized_normalized = torch.tensor(
        scaler.fit_transform(train_images_tensor_resized)
    )
    val_features_resized_normalized = torch.tensor(
        scaler.transform(validation_images_tensor_resized)
    )
    test_features_resized_normalized = torch.tensor(
        scaler.transform(test_images_tensor_resized)
    )

    train_features_normalized = train_features_normalized.float()
    val_features_normalized = val_features_normalized.float()
    test_features_normalized = test_features_normalized.float()

    train_features_resized_normalized = train_features_resized_normalized.float()
    val_features_resized_normalized = val_features_resized_normalized.float()
    test_features_resized_normalized = test_features_resized_normalized.float()





    def compute_haar_features(image):
        if image.shape != (20, 20) and image.shape != (28, 28):
            raise ValueError("Input image must be of shape 20x20 or 28x28.")

        features = []

        # Sliding window
        for i in range(0, image.shape[0], 3):  # Slide vertically with a step of 3
            for j in range(0, image.shape[0], 3):  # Slide horizontally with a step of 3

                if i + 6 > image.shape[0] or j + 6 > image.shape[0]:
                    continue

                # Extract 6x6 window
                window = image[i : i + 6, j : j + 6]

                # Horizontal feature
                horizontal_feature_value = np.sum(window[0:3, :]) - np.sum(window[3:6, :])

                # Vertical feature
                vertical_feature_value = np.sum(window[:, 0:3]) - np.sum(window[:, 3:6])

                features.append(horizontal_feature_value)
                features.append(vertical_feature_value)

        return np.array(features)


    def aspect_ratio(image, threshold=0.5):
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


    def num_regions_below_threshold(image, threshold=0.5):
        # Threshold the image so that pixels below the threshold are set to 1
        # and those above the threshold are set to 0.
        bin_image = image < threshold

        # Use connected components labeling
        labeled_array, num_features = label(bin_image)

        # Return the number of unique regions
        # (subtracting 1 as one of the labels will be the background)
        return num_features

    # compute datasets

    aspect_ratio_train = np.zeros(num_train)
    aspect_ratio_val = np.zeros(num_val)
    aspect_ratio_test = np.zeros(num_test)

    num_white_regions_train = np.zeros(num_train)
    num_white_regions_val = np.zeros(num_val)
    num_white_regions_test = np.zeros(num_test)

    for i in range(num_train):
        aspect_ratio_train[i] = aspect_ratio(train_images_tensor[i].reshape(28, 28).numpy())
        num_white_regions_train[i] = num_regions_below_threshold(
            train_images_tensor[i].reshape(28, 28)
        )

    for i in range(num_val):
        aspect_ratio_val[i] = aspect_ratio(
            validation_images_tensor[i].reshape(28, 28).numpy()
        )
        num_white_regions_val[i] = num_regions_below_threshold(
            validation_images_tensor[i].reshape(28, 28)
        )

    for i in range(num_test):
        aspect_ratio_test[i] = aspect_ratio(test_images_tensor[i].reshape(28, 28).numpy())
        num_white_regions_test[i] = num_regions_below_threshold(
            test_images_tensor[i].reshape(28, 28)
        )

    # compute datasets

    haar_1 = compute_haar_features(train_images_tensor[0].reshape(28, 28).numpy())
    len_haar_features = len(haar_1)

    features_train = np.zeros((num_train, len_haar_features + 2))
    features_val = np.zeros((num_val, len_haar_features + 2))
    features_test = np.zeros((num_test, len_haar_features + 2))

    haar_1 = compute_haar_features(train_images_tensor_resized[0].reshape(20, 20))
    len_haar_features_resized = len(haar_1)

    features_train_resized = np.zeros((num_train, len_haar_features_resized + 2))
    features_val_resized = np.zeros((num_val, len_haar_features_resized + 2))
    features_test_resized = np.zeros((num_test, len_haar_features_resized + 2))

    for i in range(num_train):
        haar_features = compute_haar_features(
            train_images_tensor[i].reshape(28, 28).numpy()
        )
        features_train[i, :] = np.hstack(
            (haar_features, aspect_ratio_train[i], num_white_regions_train[i])
        )

    for i in range(num_val):
        haar_features = compute_haar_features(
            validation_images_tensor[i].reshape(28, 28).numpy()
        )
        features_val[i, :] = np.hstack(
            (haar_features, aspect_ratio_val[i], num_white_regions_val[i])
        )

    for i in range(num_test):
        haar_features = compute_haar_features(test_images_tensor[i].reshape(28, 28).numpy())
        features_test[i, :] = np.hstack(
            (haar_features, aspect_ratio_test[i], num_white_regions_test[i])
        )

    for i in range(num_train):
        haar_features = compute_haar_features(
            train_images_tensor_resized[i].reshape(20, 20)
        )
        features_train_resized[i, :] = np.hstack(
            (haar_features, aspect_ratio_train[i], num_white_regions_train[i])
        )

    for i in range(num_val):
        haar_features = compute_haar_features(
            validation_images_tensor_resized[i].reshape(20, 20)
        )
        features_val_resized[i, :] = np.hstack(
            (haar_features, aspect_ratio_val[i], num_white_regions_val[i])
        )

    for i in range(num_test):
        haar_features = compute_haar_features(test_images_tensor_resized[i].reshape(20, 20))
        features_test_resized[i, :] = np.hstack(
            (haar_features, aspect_ratio_test[i], num_white_regions_test[i])
        )

    train_features_normalized = torch.tensor(scaler.fit_transform(features_train))
    val_features_normalized = torch.tensor(scaler.transform(features_val))
    test_features_normalized = torch.tensor(scaler.transform(features_test))

    train_features_resized_normalized = torch.tensor(
        scaler.fit_transform(features_train_resized)
    )
    val_features_resized_normalized = torch.tensor(scaler.transform(features_val_resized))
    test_features_resized_normalized = torch.tensor(scaler.transform(features_test_resized))

    train_features_normalized = train_features_normalized.float()
    val_features_normalized = val_features_normalized.float()
    test_features_normalized = test_features_normalized.float()

    train_features_resized_normalized = train_features_resized_normalized.float()
    val_features_resized_normalized = val_features_resized_normalized.float()
    test_features_resized_normalized = test_features_resized_normalized.float()

    return train_features_resized_normalized, val_features_resized_normalized, test_features_resized_normalized, train_labels_tensor, validation_labels_tensor, test_labels



def prepare_MNIST_MLP(train_features_resized_normalized, val_features_resized_normalized, test_features_resized_normalized, train_labels_tensor, validation_labels_tensor, test_labels, hidden_neuron_specification=None, prune=True):


    def evaluate_model(model):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            
            test_outputs = model(test_features_resized_normalized)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = accuracy_score(test_labels, predicted.numpy())
            print("Accuracy:", accuracy)
            return accuracy


    # Hyperparameters
    input_dim = train_features_resized_normalized.shape[1]
    output_dim = len(set(train_labels_tensor.numpy()))  # Assuming train_labels are class indices
    if(hidden_neuron_specification is None):
        hidden_dim = int((input_dim + output_dim) / 2)
    else:
        hidden_dim = hidden_neuron_specification

    # Instantiate the model
    model_medium2_resized = SimpleNN(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_medium2_resized.parameters(), lr=0.001)

    # Training loop with L1 regularization
    lambda_l1 = 0.0001  # L1 regularization coefficient

    validation_losses = []
    epoch = 0

    model_states = []

    while True:
        optimizer.zero_grad()
        outputs = model_medium2_resized(train_features_resized_normalized)

        loss = criterion(outputs, train_labels_tensor)

        # Add L1 regularization
        l1_reg = torch.tensor(0.0, requires_grad=True)
        for param in model_medium2_resized.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        loss += lambda_l1 * l1_reg

        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}, validation loss: {validation_losses[-1]:.4f}"
            )

        # store model state
        model_states.append(model_medium2_resized.state_dict())

        # Compute validation loss
        with torch.no_grad():
            outputs = model_medium2_resized(val_features_resized_normalized)
            loss = criterion(outputs, validation_labels_tensor)
            validation_losses.append(loss.item())

        # Check for early stopping if no improvement in validation loss in last 10 epochs
        if epoch > 10 and validation_losses[-1] > validation_losses[-10]:
            print("Early stopping")
            break

        epoch += 1

    best_model_state = model_states[np.argmin(validation_losses)]
    model_medium2_resized.load_state_dict(best_model_state)

    evaluate_model(model_medium2_resized)

    model_medium2_resized_pruned = copy.deepcopy(model_medium2_resized)
    if(prune):
        model_medium2_resized_pruned = prune_pytorch_network(
            model_medium2_resized_pruned, 1e-1, 1e-1
        )

    evaluate_model(model_medium2_resized_pruned)


    from sklearn.neural_network import MLPClassifier


    def pytorch_to_sklearn(pytorch_model):

        # Extract weights and biases from PyTorch model
        fc1_weight = pytorch_model.fc1.weight.data
        fc1_bias = pytorch_model.fc1.bias.data
        fc2_weight = pytorch_model.fc2.weight.data
        fc2_bias = pytorch_model.fc2.bias.data

        # Get the sizes for initialization
        input_size = fc1_weight.shape[1]
        hidden_size = fc1_weight.shape[0]
        output_size = fc2_weight.shape[0]

        # Initialize sklearn MLP
        sklearn_mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_size,), activation="relu", max_iter=1
        )

        # To ensure the model doesn't change the weights during the dummy fit, we set warm_start=True
        sklearn_mlp.warm_start = True

        # Dummy fit to initialize weights (necessary step before setting the weights)
        sklearn_mlp.fit(np.zeros((output_size, input_size)), list(range(output_size)))

        # Set the weights and biases
        sklearn_mlp.coefs_[0] = fc1_weight.t().numpy()
        sklearn_mlp.intercepts_[0] = fc1_bias.numpy()
        sklearn_mlp.coefs_[1] = fc2_weight.t().numpy()
        sklearn_mlp.intercepts_[1] = fc2_bias.numpy()

        return sklearn_mlp

    clf = pytorch_to_sklearn(model_medium2_resized_pruned)
    return clf#, train_features_resized_normalized, test_features_resized_normalized
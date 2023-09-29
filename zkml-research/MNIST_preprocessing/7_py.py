# -*- coding: utf-8 -*-
# noqa: D100
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier

with open(
    os.path.join(
        os.getcwd(), "zkml-research/MNIST_preprocessing", "sklearn_mlp_initialized.pkl"
    ),
    "rb",
) as f:
    mlp_network = pickle.load(f)


# Define the deeper PyTorch neural network
class SimpleNN(nn.Module):  # noqa: D101
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # noqa: D102
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# load the pytorch model from the stored state dict in pytorch_mlp_trained.pt
model_medium2_resized_pruned = SimpleNN(52, 31, 10)
model_medium2_resized_pruned.load_state_dict(
    torch.load(
        os.path.join(
            os.getcwd(), "zkml-research/MNIST_preprocessing", "pytorch_mlp_trained.pt"
        )
    )
)


a = 0


def pytorch_to_sklearn(pytorch_model):  # noqa: D103
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

    # To ensure the model doesn't change the weights
    # during the dummy fit, we set warm_start=True
    sklearn_mlp.warm_start = True

    # Dummy fit to initialize weights (necessary step before setting the weights)
    sklearn_mlp.fit(np.zeros((output_size, input_size)), list(range(output_size)))

    # Set the weights and biases
    sklearn_mlp.coefs_[0] = fc1_weight.t().numpy()
    sklearn_mlp.intercepts_[0] = fc1_bias.numpy()
    sklearn_mlp.coefs_[1] = fc2_weight.t().numpy()
    sklearn_mlp.intercepts_[1] = fc2_bias

    return sklearn_mlp


# Convert the example PyTorch MLP to sklearn MLP
example_sklearn_mlp = pytorch_to_sklearn(model_medium2_resized_pruned, mlp_network)


a = 0

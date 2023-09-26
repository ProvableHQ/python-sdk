# -*- coding: utf-8 -*-
# noqa: D100


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

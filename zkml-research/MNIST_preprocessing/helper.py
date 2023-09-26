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

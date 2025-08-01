from sklearn import tree
from sklearn.utils.multiclass import unique_labels

def plot_mlp_architecture(clf, ax=None):
    """
    Plot the architecture of a Multi-layer Perceptron (MLP) classifier or regressor.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    n_layers = clf.n_layers_
    layer_sizes = [clf.coefs_[0].shape[0]] + [w.shape[1] for w in clf.coefs_]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    # Draw nodes
    y_offset = 0
    for i, n in enumerate(layer_sizes):
        x = i * 2
        for j in range(n):
            ax.add_patch(mpatches.Circle((x, j - n/2), 0.2, color='skyblue', ec='k'))
        if i == 0:
            ax.text(x, n/2 + 0.5, "Input\n({})".format(n), ha='center')
        elif i == len(layer_sizes) - 1:
            ax.text(x, n/2 + 0.5, "Output\n({})".format(n), ha='center')
        else:
            ax.text(x, n/2 + 0.5, "Hidden\n({})".format(n), ha='center')
    # Draw connections
    for i in range(len(layer_sizes) - 1):
        x0, x1 = i * 2, (i + 1) * 2
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                ax.plot([x0, x1], [j - layer_sizes[i]/2, k - layer_sizes[i+1]/2], color='gray', lw=0.5, alpha=0.5)
    ax.axis('off')
    ax.set_title("MLP Architecture")
    plt.show()

def summarize_mlp(clf):
    """
    Print the architecture (layer sizes) and total number of parameters
    for an sklearn MLPClassifier or MLPRegressor.
    
    Parameters
    ----------
    clf : object
        A fitted sklearn.neural_network.MLPClassifier or MLPRegressor.
    """
    # Reconstruct full layer sizes (input, hidden…, output)
    layer_sizes = (
        [clf.coefs_[0].shape[0]]
        + list(clf.hidden_layer_sizes)
        + [clf.coefs_[-1].shape[1]]
    )
    
    # Print architecture
    print("Layer sizes (including input and output):", layer_sizes)
    print(f" ➔ Total layers (including input layer): {len(layer_sizes)}")
    print(f" ➔ Hidden layers: {len(clf.hidden_layer_sizes)}")
    print(" ➔ Output layer: 1")
    
    # Compute total parameters (weights + biases)
    total_params = sum(
        w.size + b.size for w, b in zip(clf.coefs_, clf.intercepts_)
    )
    print(f"Total parameters: {total_params:,}")
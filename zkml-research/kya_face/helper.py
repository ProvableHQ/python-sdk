from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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


    
def plot_confusion_matrix(y_true, y_pred, *,
                          labels=None,
                          normalize=None,
                          ax=None,
                          title="Confusion matrix",
                          cmap="Blues"):
    """
    Plot a confusion matrix for classification results.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels (order on both axes). If None, uses the union of labels
        present in y_true and y_pred.
    normalize : {'true', 'pred', 'all'}, default None
        Normalization mode passed to sklearn.metrics.confusion_matrix.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure/axes is created.
    title : str, default "Confusion matrix"
        Title shown above the plot.
    cmap : str or matplotlib Colormap, default "Blues"
        Colormap used for the heat-map.

    Returns
    -------
    numpy.ndarray
        The underlying confusion-matrix array (for further inspection if needed).
    """
    if labels is None:
        labels = unique_labels(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
import sklearn

def _transpile_model(model):
    """
    Transpile a model to Leo.

    Parameters
    ----------
    model : object
        The model to transpile.

    Returns
    -------
    transpilation_result : str
        The transpiled model.
    """

    # check if model is supported
    if not isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
        raise ValueError("Model is not supported.")
    
    if(isinstance(model, sklearn.tree._classes.DecisionTreeClassifier)):
        tree = model.tree_
        return _transpile_decision_tree_to_pseudocode(tree)
    
def _transpile_decision_tree_to_pseudocode(tree, feature_names=None, node=0, indentation=""):
    if(feature_names is None):
        feature_names = [f"x{i}" for i in range(tree.n_features)]
    
    left_child = tree.children_left[node]
    right_child = tree.children_right[node]

    # Base case: leaf node
    if left_child == right_child:  # means it's a leaf
        return indentation + f"Return {tree.value[node].argmax()}\n"

    # Recursive case: internal node
    feature = feature_names[tree.feature[node]]
    threshold = tree.threshold[node]

    if node == 0:
        pseudocode = f"IF {feature} <= {threshold:.2f} THEN\n"
    else:
        pseudocode = indentation + f"IF {feature} <= {threshold:.2f} THEN\n"
    
    pseudocode += _transpile_decision_tree_to_pseudocode(tree, feature_names, left_child, indentation + "    ")
    pseudocode += indentation + "ELSE\n"
    pseudocode += _transpile_decision_tree_to_pseudocode(tree, feature_names, right_child, indentation + "    ")
    return pseudocode
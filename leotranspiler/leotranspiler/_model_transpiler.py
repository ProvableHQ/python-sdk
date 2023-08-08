import sklearn

def _get_model_transpiler(model, validation_data=None):
    if(isinstance(model, sklearn.tree._classes.DecisionTreeClassifier)):
        return _DecisionTreeTranspiler(model, validation_data)
    else:
        raise ValueError("Model is not supported.")

class _ModelTranspilerBase:
    def __init__(self, model, validation_data=None):
        self.model = model
        self.validation_data = validation_data
    
    def transpile(self):
        raise NotImplementedError("This method is not implemented.")
    
    def _get_leo_type(self):
        self._get_numeric_range()
        # Todo return type based on numeric range
        return "i32"
    
    def _get_numeric_range(self):
        raise NotImplementedError("This method is not implemented.")
    
class _DecisionTreeTranspiler(_ModelTranspilerBase):
    def __init__(self, model, validation_data=None):
        super().__init__(model, validation_data)
    
    def transpile(self):
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
        
        tree = self.model.tree_
        return self._transpile_decision_tree_to_pseudocode(tree)
        
    def _transpile_decision_tree_to_pseudocode(self, tree, feature_names=None, node=0, indentation=""):
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
        
        pseudocode += self._transpile_decision_tree_to_pseudocode(tree, feature_names, left_child, indentation + "    ")
        pseudocode += indentation + "ELSE\n"
        pseudocode += self._transpile_decision_tree_to_pseudocode(tree, feature_names, right_child, indentation + "    ")
        return pseudocode
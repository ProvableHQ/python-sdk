from ._helper import _get_rounding_decimal_places
import sklearn, math

def _get_model_transpiler(model, validation_data):
    if(isinstance(model, sklearn.tree._classes.DecisionTreeClassifier)):
        return _DecisionTreeTranspiler(model, validation_data)
    else:
        raise ValueError("Model is not supported.")

class _ModelTranspilerBase:
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data
    
    def transpile(self):
        raise NotImplementedError("This method is not implemented.")
    
    def _get_leo_type(self):
        minimum_model, maximum_model = self._get_numeric_range_model()
        minimum = minimum_model
        maximum = maximum_model
        if(self.validation_data is not None):
            minimum_data, maximum_data = self._get_numeric_range_data()
            minimum = min(minimum, minimum_data)
            maximum = max(maximum, maximum_data)

        # Automatic fixed point conversion
        max_decimal_places_data = self._get_max_decimal_places_data()
        max_decimal_places_model = self._get_max_decimal_places_model()
        max_decimal_places = max(max_decimal_places_data, max_decimal_places_model)

        min_decimal_value = 10**(-max_decimal_places)
        fixed_point_min_scaling_exponent = math.log2(1 / min_decimal_value)

        fixed_point_scaling_exponent = math.ceil(fixed_point_min_scaling_exponent)
        fixed_point_scaling_factor = 2**fixed_point_scaling_exponent

        # Todo return type based on numeric range
        return "i32"
    
    def _get_numeric_range_model(self):
        raise NotImplementedError("This method is not implemented.")

    def _get_numeric_range_data(self):
        return self.validation_data.min(), self.validation_data.max()
    
    def _get_max_decimal_places_model(self):
        raise NotImplementedError("This method is not implemented.")
    
    def _get_max_decimal_places_data(self):
        return max([_get_rounding_decimal_places(val) for val in self.validation_data.ravel()])
    
class _DecisionTreeTranspiler(_ModelTranspilerBase):
    def __init__(self, model, validation_data):
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
    
    def _get_numeric_range_model(self):
        thresholds = self.model.tree_.threshold
        minimum = min(thresholds)
        maximum = max(thresholds)

        classes = self.model.classes_
        # check if classes are numeric
        if(isinstance(classes[0], int)):
            minimum = min(minimum, min(classes))
            maximum = max(maximum, max(classes))
        
        return minimum, maximum
    
    def _get_max_decimal_places_model(self):
        max_decimal_places = max([_get_rounding_decimal_places(val) for val in self.model.tree_.threshold.ravel()])
        return max_decimal_places
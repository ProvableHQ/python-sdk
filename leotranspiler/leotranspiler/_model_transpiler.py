from ._helper import _get_rounding_decimal_places
from ._leo_helper import _get_leo_integer_type
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
    
    def _numbers_get_leo_type_and_fixed_point_scaling_factor(self):
        minimum_model, maximum_model = self._get_numeric_range_model()
        minimum = minimum_model
        maximum = maximum_model
        if(self.validation_data is not None):
            minimum_data, maximum_data = self._get_numeric_range_data()
            minimum = min(minimum, minimum_data)
            maximum = max(maximum, maximum_data)

        bits_for_integer_part = math.ceil(math.log2(max(abs(minimum), abs(maximum))))
        signed_type_needed = minimum < 0

        # Fixed point parametrization
        max_decimal_places_model = self._get_max_decimal_places_model()
        max_decimal_places_data = 0
        if(self.validation_data is not None):
            max_decimal_places_data = self._get_max_decimal_places_data()
        max_decimal_places = max(max_decimal_places_data, max_decimal_places_model)

        min_decimal_value = 10**(-max_decimal_places)
        fixed_point_min_scaling_exponent = math.log2(1 / min_decimal_value)
        bits_for_fractional_part = math.ceil(fixed_point_min_scaling_exponent)
        fixed_point_scaling_factor = 2**bits_for_fractional_part

        leo_type = _get_leo_integer_type(signed_type_needed, bits_for_integer_part+bits_for_fractional_part)

        self.leo_type = leo_type
        self.fixed_point_scaling_factor = fixed_point_scaling_factor

        return leo_type, fixed_point_scaling_factor
    
    def _get_numeric_range_model(self):
        raise NotImplementedError("This method is not implemented.")

    def _get_numeric_range_data(self):
        return self.validation_data.min(), self.validation_data.max()
    
    def _get_max_decimal_places_model(self):
        raise NotImplementedError("This method is not implemented.")
    
    def _get_max_decimal_places_data(self):
        return max([_get_rounding_decimal_places(val) for val in self.validation_data.ravel()])
    
    def _convert_to_fixed_point(self, value):
        return int(round(value * self.fixed_point_scaling_factor))
    
    def _get_fixed_point_and_leo_type(self, value):
        return str(self._convert_to_fixed_point(value)) + self.leo_type
    
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
        feature_names = [f"x{i}" for i in range(tree.n_features)]
        self.feature_usage = {feature_name: False for feature_name in feature_names}

        return self._transpile_decision_tree_to_pseudocode(tree, feature_names)
        
    def _transpile_decision_tree_to_pseudocode(self, tree, feature_names, node=0, indentation=""):
        
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        # Base case: leaf node
        if left_child == right_child:  # means it's a leaf
            return indentation + f"return {self._get_fixed_point_and_leo_type(tree.value[node].argmax())};\n"

        # Recursive case: internal node
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        if node == 0:
            pseudocode = f"if {feature} <= {self._get_fixed_point_and_leo_type(threshold)}; {{\n"
        else:
            pseudocode = indentation + f"if {feature} <= {self._get_fixed_point_and_leo_type(threshold)}; {{\n"
        
        self.feature_usage[feature] = True

        pseudocode += self._transpile_decision_tree_to_pseudocode(tree, feature_names, left_child, indentation + "    ")
        pseudocode += indentation + f"}}\n"
        pseudocode += indentation + "else {\n"
        pseudocode += self._transpile_decision_tree_to_pseudocode(tree, feature_names, right_child, indentation + "    ")
        pseudocode += indentation + "}\n"
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
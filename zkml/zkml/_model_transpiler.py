# -*- coding: utf-8 -*-
import logging
import math

import numpy as np
import pandas as pd
import sklearn

from ._helper import _get_rounding_decimal_places
from ._input_generator import _InputGenerator
from ._leo_helper import _get_leo_integer_type


def _get_model_transpiler(model, validation_data):
    if isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
        return _DecisionTreeTranspiler(model, validation_data)
    else:
        raise ValueError("Model is not supported.")


class _ModelTranspilerBase:
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data

    def transpile(self, project_name, model_as_input):
        raise NotImplementedError("This method is not implemented.")

    def _numbers_get_leo_type_and_fixed_point_scaling_factor(self):
        minimum_model, maximum_model = self._get_numeric_range_model()
        minimum = minimum_model
        maximum = maximum_model
        if self.validation_data is not None:
            minimum_data, maximum_data = self._get_numeric_range_data()
            minimum = min(minimum, minimum_data)
            maximum = max(maximum, maximum_data)

        bits_for_integer_part = math.ceil(math.log2(max(abs(minimum), abs(maximum))))
        signed_type_needed = minimum < 0

        # Fixed point parametrization
        max_decimal_places_model = self._get_max_decimal_places_model()
        max_decimal_places_data = 0
        if self.validation_data is not None:
            max_decimal_places_data = self._get_max_decimal_places_data()
        max_decimal_places = max(max_decimal_places_data, max_decimal_places_model)

        min_decimal_value = 10 ** (-max_decimal_places)
        fixed_point_min_scaling_exponent = math.log2(1 / min_decimal_value)
        bits_for_fractional_part = math.ceil(fixed_point_min_scaling_exponent)
        fixed_point_scaling_factor = 2**bits_for_fractional_part

        leo_type = _get_leo_integer_type(
            signed_type_needed, bits_for_integer_part + bits_for_fractional_part
        )

        self.leo_type = leo_type
        self.fixed_point_scaling_factor = fixed_point_scaling_factor

        logging.info(
            f"Minimum number: {minimum}, maximum number: {maximum}. Recommended "
            f"fixed-point scaling factor: {fixed_point_scaling_factor}, required Leo "
            f"type: {leo_type}"
        )

        return leo_type, fixed_point_scaling_factor

    def _get_numeric_range_model(self):
        raise NotImplementedError("This method is not implemented.")

    def _get_numeric_range_data(self):
        if isinstance(self.validation_data, np.ndarray):
            return self.validation_data.min(), self.validation_data.max()
        elif isinstance(self.validation_data, pd.DataFrame):
            return self.validation_data.min().min(), self.validation_data.max().max()
        else:
            raise TypeError("Unsupported data type for validation_data")

    def _get_max_decimal_places_model(self):
        raise NotImplementedError("This method is not implemented.")

    def _get_max_decimal_places_data(self):
        if isinstance(self.validation_data, np.ndarray):
            return max(
                [
                    _get_rounding_decimal_places(val)
                    for val in self.validation_data.ravel()
                ]
            )
        elif isinstance(self.validation_data, pd.DataFrame):
            return max(
                [
                    _get_rounding_decimal_places(val)
                    for val in self.validation_data.to_numpy().ravel()
                ]
            )
        else:
            raise TypeError("Unsupported data type for validation_data")

    def _convert_to_fixed_point(self, value):
        if hasattr(value, "shape"):  # check if value is a numpy array
            vectorized_int = np.vectorize(int)
            return vectorized_int(
                value.astype(object) * self.fixed_point_scaling_factor
            )
        else:
            return int(round(value * self.fixed_point_scaling_factor))

    def convert_computation_base_outputs_to_decimal(self, computation_base):
        computation_base.fixed_point_scaling_factor = self.fixed_point_scaling_factor
        computation_base.output_decimal = self._convert_from_fixed_point(
            computation_base.output
        )

    def _convert_from_fixed_point(self, value):
        if isinstance(value, list):
            return [self._convert_from_fixed_point(val) for val in value]
        else:
            return value / self.fixed_point_scaling_factor

    def _get_fixed_point_and_leo_type(self, value):
        return str(self._convert_to_fixed_point(value)) + self.leo_type

    def _merge_into_transpiled_code(
        self,
        project_name,
        struct_definitions,
        circuit_inputs,
        circuit_outputs,
        model_logic_snippets,
    ):
        code = ""
        code += """// This file was automatically generated by the """
        code += f"""zkml LeoTranspiler.
program {project_name}.aleo {{
    {struct_definitions}
    transition main {circuit_inputs} -> {circuit_outputs} {{\n"""

        for element in model_logic_snippets:
            if isinstance(element, str):
                code += f"""{element}"""
            elif isinstance(element, _InputGenerator._Input):
                code += f"""{element.reference_name}"""
            else:
                raise ValueError("Unknown element type in model logic snippets.")

        code += """    }
}"""
        return code

    def generate_input(self, features):
        fixed_point_features = self._convert_to_fixed_point(features)
        return self.input_generator.generate_input(fixed_point_features)


class _DecisionTreeTranspiler(_ModelTranspilerBase):
    def __init__(self, model, validation_data):
        super().__init__(model, validation_data)

    def transpile(self, project_name: str, model_as_input: bool):
        """
        Transpile a model to Leo.

        Parameters
        ----------
        project_name : str
            The name of the project.
        model_as_input : bool
            Whether the model parameters should be an input to the circuit.

        Returns
        -------
        transpilation_result : str
            The transpiled model.
        """
        tree = self.model.tree_

        # Input generation
        self.input_generator = _InputGenerator()
        for _ in range(tree.n_features):
            self.input_generator.add_input(self.leo_type, "xi")

        decision_tree_logic_snippets = self._transpile_decision_tree_logic_to_leo_code(
            tree, model_as_input, indentation="        "
        )

        (
            struct_definitions,
            input_string,
            active_input_count,
        ) = self.input_generator.get_struct_definitions_and_circuit_input_string()
        circuit_inputs = "(" + input_string + ")"
        circuit_outputs = (
            "(" + self.leo_type + ")"
        )  # Todo check multi output decision trees and models

        transpilation_result = self._merge_into_transpiled_code(
            project_name,
            struct_definitions,
            circuit_inputs,
            circuit_outputs,
            decision_tree_logic_snippets,
        )
        self.active_input_count = active_input_count
        return transpilation_result

    def _transpile_decision_tree_logic_to_leo_code(
        self, tree, model_as_input, node=0, indentation=""
    ):
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        # Base case: leaf node
        if left_child == right_child:  # means it's a leaf
            return [
                indentation + f"return "
                f"{self._get_fixed_point_and_leo_type(tree.value[node].argmax())};\n"
            ]

        # Recursive case: internal node
        feature = self.input_generator.use_input(tree.feature[node])
        threshold = self._convert_to_fixed_point(tree.threshold[node])

        leo_code_snippets = []

        if node == 0:
            leo_code_snippets += [indentation + "if ", feature, " <= "]
            if model_as_input:
                leo_code_snippets += [
                    self.input_generator.add_input(
                        self.leo_type, "customi", True, threshold, "threshold"
                    )
                ]
            else:
                leo_code_snippets += [f"{threshold}{self.leo_type}"]
            leo_code_snippets += [
                " {\n",
            ]
        else:
            leo_code_snippets += [indentation + "if ", feature, " <= "]
            if model_as_input:
                leo_code_snippets += [
                    self.input_generator.add_input(
                        self.leo_type, "customi", True, threshold, "threshold"
                    )
                ]
            else:
                leo_code_snippets += [f"{threshold}{self.leo_type}"]
            leo_code_snippets += [
                " {\n",
            ]

        leo_code_snippets += self._transpile_decision_tree_logic_to_leo_code(
            tree, model_as_input, left_child, indentation + "    "
        )
        leo_code_snippets += [indentation + "}\n" + indentation + "else {\n"]

        leo_code_snippets += self._transpile_decision_tree_logic_to_leo_code(
            tree, model_as_input, right_child, indentation + "    "
        )
        leo_code_snippets += [indentation + "}\n"]
        return leo_code_snippets

    def _get_numeric_range_model(self):
        thresholds = self.model.tree_.threshold
        minimum = min(thresholds)
        maximum = max(thresholds)

        classes = self.model.classes_
        # check if classes are numeric
        if isinstance(classes[0], int):
            minimum = min(minimum, min(classes))
            maximum = max(maximum, max(classes))

        return minimum, maximum

    def _get_max_decimal_places_model(self):
        max_decimal_places = max(
            [
                _get_rounding_decimal_places(val)
                for val in self.model.tree_.threshold.ravel()
            ]
        )
        return max_decimal_places

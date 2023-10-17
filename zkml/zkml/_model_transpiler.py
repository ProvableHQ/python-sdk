# -*- coding: utf-8 -*-
import logging
import math
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.neural_network import MLPClassifier

from ._helper import _get_rounding_decimal_places
from ._input_generator import _InputGenerator
from ._leo_helper import _get_leo_integer_type


def _get_model_transpiler(model, validation_data, fixed_point_scaling_factor=None, **kwargs):
    if isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
        return _DecisionTreeTranspiler(model, validation_data, fixed_point_scaling_factor)
    elif isinstance(
        model, sklearn.neural_network._multilayer_perceptron.MLPClassifier
    ) or isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPRegressor):
        # ensure that the model uses the ReLU activation function
        if model.activation != "relu":
            raise ValueError(
                "The model uses the activation function "
                f"{model.activation}, but only ReLU is supported."
            )
        return _MLPTranspiler(model, validation_data, fixed_point_scaling_factor, **kwargs)
    else:
        raise ValueError("Model is not supported.")


class _ModelTranspilerBase:
    def __init__(self, model, validation_data, pre_set_fixed_point_scaling_factor):
        self.model = model
        self.validation_data = validation_data
        self.pre_set_fixed_point_scaling_factor = pre_set_fixed_point_scaling_factor

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

            (
                minimum_model_inference,
                maximum_model_inference,
            ) = self._get_numeric_range_model_inference()

            if minimum_model_inference is not None:
                minimum = min(minimum, minimum_model_inference)
            if maximum_model_inference is not None:
                maximum = max(maximum, maximum_model_inference)

        bits_for_integer_part = math.ceil(math.log2(max(abs(minimum), abs(maximum))))
        signed_type_needed = minimum < 0

        # Fixed point parametrization
        # todo implement this for model inference
        max_decimal_places_model = self._get_max_decimal_places_model()
        max_decimal_places_data = 0
        if self.validation_data is not None:
            max_decimal_places_data = self._get_max_decimal_places_data()
        max_decimal_places = max(max_decimal_places_data, max_decimal_places_model)

        min_decimal_value = 10 ** (-max_decimal_places)
        fixed_point_min_scaling_exponent = math.log2(1 / min_decimal_value)
        bits_for_fractional_part = math.ceil(fixed_point_min_scaling_exponent)

        if(self.pre_set_fixed_point_scaling_factor is None):
            fixed_point_scaling_factor = 2**bits_for_fractional_part
        else:
            fixed_point_scaling_factor = self.pre_set_fixed_point_scaling_factor

        if self.validation_data is not None:
            (
                minimum_model_inference_fixed_point,
                maximum_model_inference_fixed_point,
            ) = self._get_numeric_range_model_inference(
                scaling_factor=fixed_point_scaling_factor
            )

            if minimum_model_inference is not None:
                bits_for_integer_part_after_fixed_point_inference = math.ceil(
                    math.log2(
                        max(
                            abs(minimum_model_inference_fixed_point),
                            abs(maximum_model_inference_fixed_point),
                        )
                    )
                )
                bits_for_integer_part = max(
                    bits_for_integer_part,
                    bits_for_integer_part_after_fixed_point_inference,
                )

        leo_type = _get_leo_integer_type(
            signed_type_needed, bits_for_integer_part + bits_for_fractional_part
        )

        self.leo_type = leo_type
        self.fixed_point_scaling_factor = fixed_point_scaling_factor
        self.output_fixed_point_scaling_factor_power = 1

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

    def _get_numeric_range_model_inference(self, scaling_factor=1):
        # has to be implemented by the subclass if the model requires it
        return None, None

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

    def _convert_to_fixed_point(self, value, power=1):
        if hasattr(value, "shape"):  # check if value is a numpy array
            vectorized_int = np.vectorize(int)
            return vectorized_int(
                value.astype(object) * (self.fixed_point_scaling_factor**power)
            )
        else:
            return int(round(value * (self.fixed_point_scaling_factor**power)))

    def convert_computation_base_outputs_to_decimal(self, computation_base):
        computation_base.fixed_point_scaling_factor = (
            self.fixed_point_scaling_factor
            ** self.output_fixed_point_scaling_factor_power
        )
        computation_base.output_decimal = self._convert_from_fixed_point(
            computation_base.output
        )

    def _convert_from_fixed_point(self, value):
        if isinstance(value, list):
            return [self._convert_from_fixed_point(val) for val in value]
        else:
            return value / (
                self.fixed_point_scaling_factor
                ** self.output_fixed_point_scaling_factor_power
            )

    def _get_fixed_point_and_leo_type(self, value):
        return str(self._convert_to_fixed_point(value)) + self.leo_type

    def _merge_into_transpiled_code(
        self,
        project_name,
        struct_definitions,
        circuit_inputs,
        circuit_outputs,
        model_logic_snippets,
        add_relu_function=False,
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

        code += """    }"""

        if add_relu_function and False:  # todo adjust to data type
            code += """
                function relu(x: field) -> field {
        let x_integer: i128 = x as i128;
        if x_integer < 0i128 {
            return 0field;
        } else {
            return x_integer as field;
        }
    }"""

        if add_relu_function and True:  # todo adjust to data type
            code += f"function relu(x: {self.leo_type}) -> {self.leo_type} {{\n"
            code += f"if x < 0{self.leo_type} {{"
            code += f"return 0{self.leo_type};"
            code += "} else {"
            code += "return x;"
            code += """
        }
    }"""
        code += """
}"""
        return code

    def generate_input(self, features):
        fixed_point_features = self._convert_to_fixed_point(features)
        return self.input_generator.generate_input(fixed_point_features)


class _DecisionTreeTranspiler(_ModelTranspilerBase):
    def __init__(self, model, validation_data, pre_set_fixed_point_scaling_factor):
        super().__init__(model, validation_data, pre_set_fixed_point_scaling_factor)

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


class _MLPTranspiler(_ModelTranspilerBase):
    def __init__(self, model, validation_data, pre_set_fixed_point_scaling_factor, **kwargs):
        super().__init__(model, validation_data, pre_set_fixed_point_scaling_factor)

        if("data_representation_type" in kwargs):
            self.data_representation_type = kwargs["data_representation_type"]
        else:
            self.data_representation_type = "int"

        if("layer_wise_fixed_point_scaling_factor" in kwargs):
            self.layer_wise_fixed_point_scaling_factor = kwargs["layer_wise_fixed_point_scaling_factor"]
        else:
            self.layer_wise_fixed_point_scaling_factor = True

    def _get_numeric_range_model(self):
        minimum = None
        maximum = None

        for layer in self.model.coefs_:
            layer_minimum = layer.min()
            layer_maximum = layer.max()
            if minimum is None:
                minimum = layer_minimum
                maximum = layer_maximum
            else:
                minimum = min(minimum, layer_minimum)
                maximum = max(maximum, layer_maximum)

        for layer in self.model.intercepts_:
            layer_minimum = layer.min()
            layer_maximum = layer.max()
            minimum = min(minimum, layer_minimum)
            maximum = max(maximum, layer_maximum)

        # check if model is a classifier
        if isinstance(self.model, MLPClassifier):
            classes = self.model.classes_
            # check if classes are numeric
            # Todo generalize this code with others like the decision trees
            # Todo one could quantize here and make the range smaller
            if isinstance(classes[0], int) or isinstance(classes[0], np.int64):
                minimum = min(minimum, min(classes))
                maximum = max(maximum, max(classes))

        return minimum, maximum

    def _get_numeric_range_model_inference(self, scaling_factor=1):
        minimum = None
        maximum = None

        for data_point in self.validation_data:
            min_inference, max_inference = self._get_min_max_pre_activation_values(
                self.model, data_point, scaling_factor
            )
            if minimum is None:
                minimum = min_inference
                maximum = max_inference
            else:
                minimum = min(minimum, min_inference)
                maximum = max(maximum, max_inference)

        return minimum, maximum

    def _get_min_max_pre_activation_values(self, model, X, scaling_factor=1):
        # todo implement rounding in fixed point number conversion
        layer_input = X
        global_min = float("inf")
        global_max = float("-inf")

        # Iterate through layers and compute weighted sum
        for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
            pre_activation = np.dot(
                layer_input * scaling_factor, weights * scaling_factor
            ) + biases * (scaling_factor ** (i + 2))

            # Update global min and max pre-activation values and layer-specific fixed point scaled bias
            global_min = min(
                global_min,
                np.min(pre_activation),
                np.min(biases * (scaling_factor ** (i + 2))),
            )
            global_max = max(
                global_max,
                np.max(pre_activation),
                np.max(biases * (scaling_factor ** (i + 2))),
            )

            # Apply ReLU activation function except for the output layer
            if i < len(model.coefs_) - 1:
                layer_input = np.maximum(0, pre_activation)

        return global_min, global_max

    def _get_max_decimal_places_model(self):
        max_decimal_places = None
        for layer in self.model.coefs_:
            layer_max_decimal_places = max(
                [_get_rounding_decimal_places(val) for val in layer.ravel()]
            )
            if max_decimal_places is None:
                max_decimal_places = layer_max_decimal_places
            else:
                max_decimal_places = max(max_decimal_places, layer_max_decimal_places)

        for layer in self.model.intercepts_:
            layer_max_decimal_places = max(
                [_get_rounding_decimal_places(val) for val in layer.ravel()]
            )
            max_decimal_places = max(max_decimal_places, layer_max_decimal_places)
        return max_decimal_places

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
        # Input generation
        number_of_model_inputs = self.model.coefs_[0].shape[0]
        self.input_generator = _InputGenerator()
        for _ in range(number_of_model_inputs):
            self.input_generator.add_input(self.leo_type, "xi")

        mlp_logic_snippets = self._transpile_mlp_logic_to_leo_code(
            self.model, model_as_input, indentation="        "
        )

        (
            struct_definitions,
            input_string,
            active_input_count,
        ) = self.input_generator.get_struct_definitions_and_circuit_input_string()
        circuit_inputs = "(" + input_string + ")"

        num_outputs = self.model.coefs_[-1].shape[1]
        circuit_outputs = (
            "(" + ", ".join([self.leo_type for _ in range(num_outputs)]) + ")"
        )

        transpilation_result = self._merge_into_transpiled_code(
            project_name,
            struct_definitions,
            circuit_inputs,
            circuit_outputs,
            mlp_logic_snippets,
            add_relu_function=True,
        )
        self.active_input_count = active_input_count

        if(False):
            pseudocode = self.mlp_to_pseudocode(self.model)
            with open(
                os.path.join(
                    os.getcwd(),
                    "zkml",
                    "tests",
                    "tmp",
                    "mnist",
                    project_name,
                    "pseudocode.txt",
                ),
                "w",
            ) as f:
                f.write(pseudocode)

        return transpilation_result

    def _transpile_mlp_logic_to_leo_code(
        self,
        mlp,
        model_as_input,
        indentation="",
        prune_threshold_weights=0,
        prune_threshold_bias=0,
    ):
        
        if (self.data_representation_type == "int" and self.layer_wise_fixed_point_scaling_factor):
            return self._transpile_mlp_logic_to_leo_code_3(
                mlp,
                model_as_input,
                indentation,
                prune_threshold_weights,
                prune_threshold_bias,
            )
        if(self.data_representation_type == "int" and not self.layer_wise_fixed_point_scaling_factor):
            return self._transpile_mlp_logic_to_leo_code_4(
                mlp,
                model_as_input,
                indentation,
                prune_threshold_weights,
                prune_threshold_bias,
            )
        if(self.data_representation_type == "field" and self.layer_wise_fixed_point_scaling_factor):
            return self._transpile_mlp_logic_to_leo_code_1(
                mlp,
                model_as_input,
                indentation,
                prune_threshold_weights,
                prune_threshold_bias,
            )
        if(self.data_representation_type == "field" and not self.layer_wise_fixed_point_scaling_factor):
            raise NotImplementedError("This method is not implemented. You can use the field representation with layer-wise fixed point scaling factor, or the integer representation with or without layer-wise fixed point scaling factor.")

    def _transpile_mlp_logic_to_leo_code_1(
        self,
        mlp,
        model_as_input,
        indentation="",
        prune_threshold_weights=0,
        prune_threshold_bias=0,
    ):
        # initial version
        leo_code_snippets = []

        coefs = mlp.coefs_
        intercepts = mlp.intercepts_

        # specify and convert inputs to fields
        for i in range(coefs[0].shape[0]):
            used_input = self.input_generator.use_input(i)

            leo_code_snippets.append(
                indentation + f"let {used_input.reference_name}_field: field = "
            )

            leo_code_snippets.append(used_input)

            leo_code_snippets.append(" as field;" + "\n")

            used_input.field_name = f"{used_input.reference_name}_field"

        # for each layer
        prev_neurons = [
            f"{self.input_generator.use_input(i).field_name}"
            for i in range(coefs[0].shape[0])
        ]
        for layer in range(len(coefs)):  # for each layer
            layer_code = []
            for n in range(coefs[layer].shape[1]):  # for each neuron in the layer
                terms = []
                for i in range(coefs[layer].shape[0]):  # for each input to the neuron
                    weight_input = self.input_generator.add_input(
                        self.leo_type,
                        "customi",
                        model_as_input,
                        coefs[layer][i][n],
                        f"w_{layer}_{n}_",
                    )
                    # todo adapt for case where model weights are actual inputs
                    if abs(coefs[layer][i][n]) > prune_threshold_weights:
                        terms.append(
                            f"({self._convert_to_fixed_point(weight_input.value.item())}{self.leo_type} as field)*{prev_neurons[i]}"
                        )

                if layer != len(coefs) - 1:  # if not the last layer
                    neuron_code = indentation + f"let neuron_{layer+1}_{n}: field = "
                    weights_or_bias_above_prune_thresholds = (
                        terms != [] or abs(intercepts[layer][n]) > prune_threshold_bias
                    )

                    if weights_or_bias_above_prune_thresholds:
                        neuron_code += "relu(" + " + ".join(terms)

                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            neuron_code += f" + ({self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type} as field)"

                        neuron_code += ");\n"
                    else:
                        neuron_code += "0field;\n"
                    leo_code_snippets.append(neuron_code)

                else:  # if the last layer
                    neuron_code = indentation + f"let output_{n}_field" + f" : field = "
                    self.output_fixed_point_scaling_factor_power = layer + 2

                    if terms != [] and abs(intercepts[layer][n]) > prune_threshold_bias:
                        neuron_code += f"{' + '.join(terms)}"
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            bias_input = self.input_generator.add_input(
                                self.leo_type,
                                "customi",
                                model_as_input,
                                intercepts[layer][n],
                                f"b_{layer}_{n}_",
                            )
                            neuron_code += f" + ({self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type} as field)"
                        neuron_code += ";\n"
                    elif (
                        terms == [] and abs(intercepts[layer][n]) > prune_threshold_bias
                    ):
                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        neuron_code += f"({self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type} as field);\n"
                    elif (
                        terms != []
                        and abs(intercepts[layer][n]) <= prune_threshold_bias
                    ):
                        neuron_code += f"{' + '.join(terms)};\n"
                    else:
                        neuron_code += "0field;\n"

                    leo_code_snippets.append(neuron_code)
            prev_neurons = [
                f"neuron_{layer+1}_{n}" for n in range(coefs[layer].shape[1])
            ]

        num_outputs = coefs[-1].shape[1]
        return_line = (
            indentation
            + "return ("
            + ", ".join(
                [f"output_{i}_field as {self.leo_type}" for i in range(num_outputs)]
            )
            + ");\n"
        )
        leo_code_snippets.append(return_line)
        return leo_code_snippets

    def _transpile_mlp_logic_to_leo_code_2(
        self,
        mlp,
        model_as_input,
        indentation="",
        prune_threshold_weights=0,
        prune_threshold_bias=0,
    ):
        # optimizations through subtracting positive integer fields instead of adding negative integer fields
        leo_code_snippets = []

        coefs = mlp.coefs_
        intercepts = mlp.intercepts_

        # specify and convert inputs to fields
        for i in range(coefs[0].shape[0]):
            used_input = self.input_generator.use_input(i)

            leo_code_snippets.append(
                indentation + f"let {used_input.reference_name}_field: field = "
            )

            leo_code_snippets.append(used_input)

            leo_code_snippets.append(" as field;" + "\n")

            used_input.field_name = f"{used_input.reference_name}_field"

        # for each layer
        prev_neurons = [
            f"{self.input_generator.use_input(i).field_name}"
            for i in range(coefs[0].shape[0])
        ]
        for layer in range(len(coefs)):  # for each layer
            layer_code = []
            for n in range(coefs[layer].shape[1]):  # for each neuron in the layer
                terms = []
                for i in range(coefs[layer].shape[0]):  # for each input to the neuron
                    weight_input = self.input_generator.add_input(
                        self.leo_type,
                        "customi",
                        model_as_input,
                        coefs[layer][i][n],
                        f"w_{layer}_{n}_",
                    )
                    # todo adapt for case where model weights are actual inputs
                    if abs(coefs[layer][i][n]) > prune_threshold_weights:
                        terms.append(
                            f"({self._convert_to_fixed_point(weight_input.value.item())}{self.leo_type} as field)*{prev_neurons[i]}"
                        )

                if layer != len(coefs) - 1:  # if not the last layer
                    neuron_code = indentation + f"let neuron_{layer+1}_{n}: field = "
                    weights_or_bias_above_prune_thresholds = (
                        terms != [] or abs(intercepts[layer][n]) > prune_threshold_bias
                    )

                    if weights_or_bias_above_prune_thresholds:
                        neuron_code += "relu("  # + " + ".join(terms)
                        for i, term in enumerate(terms):
                            if i == 0:
                                if term[1] == "-":
                                    neuron_code += f" ({term[2:]}"
                                else:
                                    neuron_code += f"{term}"
                            else:
                                if term[1] == "-":
                                    neuron_code += f" - ({term[2:]}"
                                else:
                                    neuron_code += f" + {term}"

                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            neuron_code += f" + ({self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type} as field)"  # todo handle negative bias separately

                        neuron_code += ");\n"
                    else:
                        neuron_code += "0field;\n"
                    leo_code_snippets.append(neuron_code)

                else:  # if the last layer
                    neuron_code = indentation + f"let output_{n}_field" + f" : field = "
                    self.output_fixed_point_scaling_factor_power = layer + 2

                    if terms != [] and abs(intercepts[layer][n]) > prune_threshold_bias:
                        # neuron_code += f"{' + '.join(terms)}"
                        for i, term in enumerate(terms):
                            if i == 0:
                                if term[1] == "-":
                                    neuron_code += f" ({term[2:]}"
                                else:
                                    neuron_code += f"{term}"
                            else:
                                if term[1] == "-":
                                    neuron_code += f" - ({term[2:]}"
                                else:
                                    neuron_code += f" + {term}"

                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            bias_input = self.input_generator.add_input(
                                self.leo_type,
                                "customi",
                                model_as_input,
                                intercepts[layer][n],
                                f"b_{layer}_{n}_",
                            )
                            neuron_code += f" + ({self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type} as field)"
                        neuron_code += ";\n"
                    elif (
                        terms == [] and abs(intercepts[layer][n]) > prune_threshold_bias
                    ):
                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        neuron_code += f"({self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type} as field);\n"
                    elif (
                        terms != []
                        and abs(intercepts[layer][n]) <= prune_threshold_bias
                    ):
                        # neuron_code += f"{' + '.join(terms)};\n"
                        for i, term in enumerate(terms):
                            if i == 0:
                                if term[1] == "-":
                                    neuron_code += f" ({term[2:]}"
                                else:
                                    neuron_code += f"{term}"
                            else:
                                if term[1] == "-":
                                    neuron_code += f" - ({term[2:]}"
                                else:
                                    neuron_code += f" + {term}"
                        neuron_code += ";\n"
                    else:
                        neuron_code += "0field;\n"

                    leo_code_snippets.append(neuron_code)
            prev_neurons = [
                f"neuron_{layer+1}_{n}" for n in range(coefs[layer].shape[1])
            ]

        num_outputs = coefs[-1].shape[1]
        return_line = (
            indentation
            + "return ("
            + ", ".join(
                [f"output_{i}_field as {self.leo_type}" for i in range(num_outputs)]
            )
            + ");\n"
        )
        leo_code_snippets.append(return_line)
        return leo_code_snippets

    def _transpile_mlp_logic_to_leo_code_3(
        self,
        mlp,
        model_as_input,
        indentation="",
        prune_threshold_weights=0,
        prune_threshold_bias=0,
    ):
        # no usage of fields, but of layer-wise fixed point scaling factors
        leo_code_snippets = []

        coefs = mlp.coefs_
        intercepts = mlp.intercepts_

        # for each layer
        prev_neurons = [
            self.input_generator.use_input(i)
            for i in range(len(self.input_generator.input_list))
        ]
        for layer in range(len(coefs)):  # for each layer
            for n in range(coefs[layer].shape[1]):  # for each neuron in the layer
                terms = []
                for i in range(coefs[layer].shape[0]):  # for each input to the neuron
                    weight_input = self.input_generator.add_input(
                        self.leo_type,
                        "customi",
                        model_as_input,
                        coefs[layer][i][n],
                        f"w_{layer}_{n}_",
                    )
                    # todo adapt for case where model weights are actual inputs
                    if abs(coefs[layer][i][n]) > prune_threshold_weights:
                        terms.append(
                            f"{self._convert_to_fixed_point(weight_input.value.item())}{self.leo_type} * "
                        )
                        terms.append(prev_neurons[i])

                if layer != len(coefs) - 1:  # if not the last layer
                    leo_code_snippets.append(
                        indentation + f"let neuron_{layer+1}_{n}: {self.leo_type} = "
                    )
                    weights_or_bias_above_prune_thresholds = (
                        terms != [] or abs(intercepts[layer][n]) > prune_threshold_bias
                    )

                    if weights_or_bias_above_prune_thresholds:
                        leo_code_snippets.append("relu(")

                        for i, term in enumerate(terms):
                            # if i greater 0 add plus sign
                            if i > 0 and i % 2 == 0:
                                leo_code_snippets.append(" + ")
                            leo_code_snippets.append(term)

                        # delete the last item if it is a plus sign
                        if leo_code_snippets[-1] == " + ":
                            leo_code_snippets.pop()

                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            leo_code_snippets.append(
                                f" + {self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type}"
                            )

                        leo_code_snippets.append(");\n")
                    else:
                        leo_code_snippets.append(f"0{self.leo_type};\n")

                else:  # if the last layer
                    neuron_code = (
                        indentation + f"let output_{n}" + f" : {self.leo_type} = "
                    )
                    self.output_fixed_point_scaling_factor_power = layer + 2

                    if terms != [] and abs(intercepts[layer][n]) > prune_threshold_bias:
                        for i, term in enumerate(terms):
                            # if i greater 0 add plus sign
                            if i > 0 and i % 2 == 0:
                                neuron_code += " + "
                            neuron_code += term
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            bias_input = self.input_generator.add_input(
                                self.leo_type,
                                "customi",
                                model_as_input,
                                intercepts[layer][n],
                                f"b_{layer}_{n}_",
                            )
                            neuron_code += f" + {self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type}"
                        neuron_code += ";\n"
                    elif (
                        terms == [] and abs(intercepts[layer][n]) > prune_threshold_bias
                    ):
                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        neuron_code += f"{self._convert_to_fixed_point(bias_input.value.item(), layer+2)}{self.leo_type};\n"
                    elif (
                        terms != []
                        and abs(intercepts[layer][n]) <= prune_threshold_bias
                    ):
                        for i, term in enumerate(terms):
                            # if i greater 0 add plus sign
                            if i > 0 and i % 2 == 0:
                                neuron_code += " + "
                            neuron_code += term
                        neuron_code += ";\n"
                    else:
                        neuron_code += f"0{self.leo_type};\n"

                    leo_code_snippets.append(neuron_code)
            prev_neurons = [
                f"neuron_{layer+1}_{n}" for n in range(coefs[layer].shape[1])
            ]

        num_outputs = coefs[-1].shape[1]
        return_line = (
            indentation
            + "return ("
            + ", ".join([f"output_{i}" for i in range(num_outputs)])
            + ");\n"
        )
        leo_code_snippets.append(return_line)
        return leo_code_snippets

    def _transpile_mlp_logic_to_leo_code_4(
        self,
        mlp,
        model_as_input,
        indentation="",
        prune_threshold_weights=0,
        prune_threshold_bias=0,
    ):
        # no usage of fields, and also no usage of layer-wise fixed point scaling factors
        leo_code_snippets = []

        coefs = mlp.coefs_
        intercepts = mlp.intercepts_

        # for each layer
        prev_neurons = [
            self.input_generator.use_input(i)
            for i in range(len(self.input_generator.input_list))
        ]
        for layer in range(len(coefs)):  # for each layer
            for n in range(coefs[layer].shape[1]):  # for each neuron in the layer
                terms = []
                for i in range(coefs[layer].shape[0]):  # for each input to the neuron
                    weight_input = self.input_generator.add_input(
                        self.leo_type,
                        "customi",
                        model_as_input,
                        coefs[layer][i][n],
                        f"w_{layer}_{n}_",
                    )
                    # todo adapt for case where model weights are actual inputs
                    if abs(coefs[layer][i][n]) > prune_threshold_weights:
                        terms.append(
                            f"{self._convert_to_fixed_point(weight_input.value.item())}{self.leo_type} * "
                        )
                        terms.append(prev_neurons[i])
                        terms.append(
                            f" / {self.fixed_point_scaling_factor}{self.leo_type}"
                        )

                if layer != len(coefs) - 1:  # if not the last layer
                    leo_code_snippets.append(
                        indentation + f"let neuron_{layer+1}_{n}: {self.leo_type} = "
                    )
                    weights_or_bias_above_prune_thresholds = (
                        terms != [] or abs(intercepts[layer][n]) > prune_threshold_bias
                    )

                    if weights_or_bias_above_prune_thresholds:
                        leo_code_snippets.append("relu(")

                        for i, term in enumerate(terms):
                            # if i greater 0 add plus sign
                            if i > 0 and i % 3 == 0:
                                leo_code_snippets.append(" + ")
                            leo_code_snippets.append(term)

                        # delete the last item if it is a plus sign
                        if leo_code_snippets[-1] == " + ":
                            leo_code_snippets.pop()

                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            leo_code_snippets.append(
                                f" + {self._convert_to_fixed_point(bias_input.value.item())}{self.leo_type}"
                            )

                        leo_code_snippets.append(");\n")
                    else:
                        leo_code_snippets.append(f"0{self.leo_type};\n")

                else:  # if the last layer
                    neuron_code = (
                        indentation + f"let output_{n}" + f" : {self.leo_type} = "
                    )

                    if terms != [] and abs(intercepts[layer][n]) > prune_threshold_bias:
                        for i, term in enumerate(terms):
                            # if i greater 0 add plus sign
                            if i > 0 and i % 3 == 0:
                                neuron_code += " + "
                            neuron_code += term
                        if abs(intercepts[layer][n]) > prune_threshold_bias:
                            bias_input = self.input_generator.add_input(
                                self.leo_type,
                                "customi",
                                model_as_input,
                                intercepts[layer][n],
                                f"b_{layer}_{n}_",
                            )
                            neuron_code += f" + {self._convert_to_fixed_point(bias_input.value.item())}{self.leo_type}"
                        neuron_code += ";\n"
                    elif (
                        terms == [] and abs(intercepts[layer][n]) > prune_threshold_bias
                    ):
                        bias_input = self.input_generator.add_input(
                            self.leo_type,
                            "customi",
                            model_as_input,
                            intercepts[layer][n],
                            f"b_{layer}_{n}_",
                        )
                        neuron_code += f"{self._convert_to_fixed_point(bias_input.value.item())}{self.leo_type};\n"
                    elif (
                        terms != []
                        and abs(intercepts[layer][n]) <= prune_threshold_bias
                    ):
                        for i, term in enumerate(terms):
                            # if i greater 0 add plus sign
                            if i > 0 and i % 3 == 0:
                                neuron_code += " + "
                            neuron_code += term
                        neuron_code += ";\n"
                    else:
                        neuron_code += f"0{self.leo_type};\n"

                    leo_code_snippets.append(neuron_code)
            prev_neurons = [
                f"neuron_{layer+1}_{n}" for n in range(coefs[layer].shape[1])
            ]

        num_outputs = coefs[-1].shape[1]
        return_line = (
            indentation
            + "return ("
            + ", ".join([f"output_{i}" for i in range(num_outputs)])
            + ");\n"
        )
        leo_code_snippets.append(return_line)
        return leo_code_snippets

    def mlp_to_pseudocode(self, mlp):
        coefs = mlp.coefs_
        intercepts = mlp.intercepts_

        code = []
        code.append(
            "function neural_network("
            + ", ".join([f"input_{i}" for i in range(coefs[0].shape[0])])
            + "):"
        )

        # for each layer
        prev_neurons = [f"input_{i}" for i in range(coefs[0].shape[0])]
        for layer in range(len(coefs)):
            layer_code = []
            for n in range(coefs[layer].shape[1]):
                terms = [
                    f"{coefs[layer][i][n]:.5f}*{prev_neurons[i]}"
                    for i in range(coefs[layer].shape[0])
                ]
                if layer != len(coefs) - 1:  # if not the last layer
                    neuron_name = f"neuron_{layer+1}_{n}"
                    layer_code.append(
                        f"    {neuron_name} = max(0, {' + '.join(terms)}"
                        f" + {intercepts[layer][n]:.5f})"
                    )
                else:  # if the last layer
                    neuron_name = f"output_{n}"
                    layer_code.append(
                        f"    {neuron_name} = {' + '.join(terms)} + "
                        f"{intercepts[layer][n]:.5f}"
                    )
            code.extend(layer_code)
            prev_neurons = [
                f"neuron_{layer+1}_{n}" for n in range(coefs[layer].shape[1])
            ]

        outputs = [f"output_{i}" for i in range(coefs[-1].shape[1])]
        code.append(f"    return softmax([{', '.join(outputs)}])")
        return "\n".join(code)

# -*- coding: utf-8 -*-
class _InputGenerator:
    class _Struct:
        def __init__(self, fields=None, hierarchy=0):
            self.fields = fields
            self.hierarchy = hierarchy

    class _Input:
        def __init__(self, value, leo_type, active, name):
            self.value = value
            self.leo_type = leo_type
            self.active = active
            self.name = name
            self.tmp_data_value = None

        def get_set_value(self):
            if self.value is None:
                return self.tmp_data_value
            else:
                return self.value

    def __init__(self):
        self.MAX_CIRCUIT_INPUTS = 16
        self.MAX_STRUCT_FIELDS = 32
        self.MAX_STRUCT_HIERARCHY = 32

        self.MAX_INPUT_VALUES = self.MAX_CIRCUIT_INPUTS * (
            self.MAX_STRUCT_FIELDS**self.MAX_STRUCT_HIERARCHY
        )
        self.MAX_INPUT_VALUES = 16
        """Todo, for now set the value to 16 because
        hierarchical structs are not implemented yet"""
        self.inputl_list = []
        self._input_counts = {}

    def add_input(
        self, leo_type, naming_strategy="xi", active=False, value=None, name=None
    ):  # Todo implement custom naming strategy for different names
        input_count = self._input_counts.get(naming_strategy, 0)
        if naming_strategy == "xi":
            name = f"x{input_count}"
        elif naming_strategy == "custom":
            if name is None:
                raise Exception("Custom naming strategy requires a name")
        elif naming_strategy == "customi":
            if name is None:
                raise Exception("Custom naming strategy requires a name")
            name = f"{name}{input_count}"
        else:
            raise Exception("Invalid naming strategy")

        self._input_counts[naming_strategy] = input_count + 1

        new_input = self._Input(value, leo_type, active, name)
        self.inputl_list.append(new_input)

        return new_input

    def use_input(self, index):
        self.inputl_list[index].active = True
        return self.inputl_list[index]

    def get_circuit_input_string(self):
        circuit_inputs_string = ""
        active_input_count = 0
        for input in self.inputl_list:
            if input.active:
                circuit_inputs_string += f"{input.name}: {input.leo_type}, "
                active_input_count += 1

        if active_input_count == 0:
            raise Exception("No active inputs")
        elif active_input_count > self.MAX_CIRCUIT_INPUTS:
            raise Exception(
                f"Too many active inputs "
                f"({active_input_count} > {self.MAX_INPUT_VALUES})"
            )

        circuit_inputs_string = circuit_inputs_string[:-2]
        return circuit_inputs_string

    def generate_input(self, fixed_point_features):
        inputs_without_value = len(
            [input for input in self.inputl_list if input.value is None]
        )
        if len(fixed_point_features) != inputs_without_value:
            raise Exception(
                f"Number of features ({len(fixed_point_features)}) "
                "does not match number of inputs without a specified value "
                f"({inputs_without_value})"
            )

        # assign values to inputs without a specified value
        index = 0
        for input in self.inputl_list:
            if input.value is None:
                input.tmp_data_value = fixed_point_features[index]
                index += 1

        # construct input string
        input_list = []
        for input in self.inputl_list:
            if input.active:
                input_list += [f"{input.get_set_value()}{input.leo_type}"]

        return input_list

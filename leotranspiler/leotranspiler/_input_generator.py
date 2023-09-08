# -*- coding: utf-8 -*-
import pandas as pd


class _InputGenerator:
    class _Struct:
        MAX_STRUCT_FIELDS = 32
        MAX_STRUCT_HIERARCHY = 32

        def __init__(self, inputs, horizontal_position, hierarchy, parent_struct=None):
            self.horizontal_position = horizontal_position
            self.hierarchy = hierarchy
            self.parent_struct = parent_struct
            self.leo_type = None
            self.internal_reference_name = None

            self.name = f"struct{self.hierarchy}_{self.horizontal_position}"
            if parent_struct is not None:
                self.reference_name = f"{parent_struct.reference_name}.{self.name}"
            else:
                self.reference_name = self.name

            num_inputs = len(inputs)
            if (
                num_inputs > self.MAX_STRUCT_FIELDS
                and hierarchy < self.MAX_STRUCT_HIERARCHY
            ):
                # split inputs into MAX_STRUCT_FIELDS chunks

                chunk_size = num_inputs // self.MAX_STRUCT_FIELDS
                remainder = num_inputs % self.MAX_STRUCT_FIELDS

                input_chunks = []
                start_index = 0

                # Create MAX_STRUCT_FIELDS chunks
                for i in range(self.MAX_STRUCT_FIELDS):
                    end_index = start_index + chunk_size + (1 if i < remainder else 0)
                    input_chunks.append(inputs[start_index:end_index])
                    start_index = end_index

                # Create a _Struct for each chunk, with incremented hierarchy level
                self.fields = [
                    self.__class__(
                        chunk, i, hierarchy=self.hierarchy + 1, parent_struct=self
                    )
                    for i, chunk in enumerate(input_chunks)
                ]
            elif (
                num_inputs <= self.MAX_STRUCT_FIELDS
                and hierarchy < self.MAX_STRUCT_HIERARCHY
            ):
                self.fields = inputs
                for i, field in enumerate(self.fields):
                    field.horizontal_position = i
                    field.parent_struct = self
            else:
                raise ValueError("Invalid number of inputs")

            if isinstance(self.fields[0], _InputGenerator._Input):
                for i, field in enumerate(self.fields):
                    field.name = f"x{i}"
                    field.reference_name = f"{self.reference_name}.{field.name}"

        def add_struct_definition_to_directory(self, unique_struct_directory):
            self.type_identifier = f"{self.hierarchy}"
            for field in self.fields:
                self.type_identifier += field.leo_type

            if self.type_identifier in unique_struct_directory:
                self.leo_type, _ = unique_struct_directory[self.type_identifier]
            else:
                num_structs = len(unique_struct_directory.keys())
                self.leo_type = f"Struct{num_structs}"
                if isinstance(self.fields[0], _InputGenerator._Struct):
                    leo_struct_definition = (
                        f"struct {self.leo_type}"
                        + " {\n"
                        + "  "
                        + ",\n  ".join(
                            [f"{field.name}: {field.leo_type}" for field in self.fields]
                        )
                        + "\n}"
                    )
                if isinstance(self.fields[0], _InputGenerator._Input):
                    leo_struct_definition = (
                        f"struct {self.leo_type}"
                        + " {\n"
                        + "  "
                        + ",\n  ".join(
                            [
                                f"x{i}: {field.leo_type}"
                                for i, field in enumerate(self.fields)
                            ]
                        )
                        + "\n}"
                    )
                unique_struct_directory[self.type_identifier] = (
                    self.leo_type,
                    leo_struct_definition,
                )

        def get_input_value_string(self):
            str = "{ "
            for field in self.fields:
                str += f"{field.name}: {field.get_input_value_string()}, "
            return str[:-2] + " }"

    class _Input:
        def __init__(self, value, leo_type, active, name):
            self.value = value
            self.leo_type = leo_type
            self.active = active
            self.name = name
            self.reference_name = name
            self.tmp_data_value = None
            self.horizontal_position = None
            self.parent_struct = None

        def get_set_value(self):
            if self.value is None:
                return self.tmp_data_value
            else:
                return self.value

        def get_input_value_string(self):
            return f"{self.get_set_value()}{self.leo_type}"

    def __init__(self):
        self.MAX_CIRCUIT_INPUTS = 16

        self.MAX_INPUT_VALUES = self.MAX_CIRCUIT_INPUTS * (
            self._Struct.MAX_STRUCT_FIELDS**self._Struct.MAX_STRUCT_HIERARCHY
        )
        self.input_list = []
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
        self.input_list.append(new_input)

        return new_input

    def use_input(self, index):
        self.input_list[index].active = True
        return self.input_list[index]

    def get_struct_definitions_and_circuit_input_string(self):
        active_input_count = self._assign_inputs_to_structs()
        struct_definitions = self.generate_struct_definitions()
        circuit_input_string = self.generate_circuit_input_string()
        return struct_definitions, circuit_input_string, active_input_count

    def _assign_inputs_to_structs(self):
        active_inputs = [input for input in self.input_list if input.active]
        active_input_count = len(active_inputs)
        if active_input_count > self.MAX_INPUT_VALUES:
            raise ValueError(
                f"Number of active inputs ({active_input_count}) exceeds maximum "
                f"number of inputs ({self.MAX_INPUT_VALUES})"
            )

        self.structured_inputs = []
        if active_input_count <= self.MAX_CIRCUIT_INPUTS:
            self.structured_inputs = active_inputs
            for i, input in enumerate(self.structured_inputs):
                input.horizontal_position = i
                input.parent_struct = None
        else:
            # split active_inputs into MAX_CIRCUIT_INPUTS sized chunks
            # Calculate the number of elements in each chunk
            chunk_size = len(active_inputs) // self.MAX_CIRCUIT_INPUTS
            remainder = len(active_inputs) % self.MAX_CIRCUIT_INPUTS

            input_chunks = []
            start_index = 0

            # Create MAX_CIRCUIT_INPUTS chunks
            for i in range(self.MAX_CIRCUIT_INPUTS):
                end_index = start_index + chunk_size + (1 if i < remainder else 0)
                input_chunks.append(active_inputs[start_index:end_index])
                start_index = end_index

            # Create a struct for each chunk
            for i, chunk in enumerate(input_chunks):
                self.structured_inputs.append(
                    self._Struct(
                        chunk, horizontal_position=i, hierarchy=0, parent_struct=None
                    )
                )

        return active_input_count

    def generate_struct_definitions(self):
        self.unique_struct_directory = {}

        if isinstance(self.structured_inputs[0], self._Input):
            return ""

        for structured_input in self.structured_inputs:
            item = structured_input
            while True:
                if isinstance(item, self._Struct):
                    # dive deep leftmost in self.structured_inputs until hitting input
                    while isinstance(item, self._Struct):
                        item = item.fields[0]
                    item = item.parent_struct.parent_struct

                if item is not None:
                    for field in item.fields:
                        field.add_struct_definition_to_directory(
                            self.unique_struct_directory
                        )

                if item is None or item.parent_struct is None:
                    structured_input.add_struct_definition_to_directory(
                        self.unique_struct_directory
                    )
                    break
                else:
                    item = item.parent_struct

        return "\n".join(
            [
                struct_definition
                for _, struct_definition in self.unique_struct_directory.values()
            ]
        )

    def generate_circuit_input_string(self):
        input_string = ""
        for structured_input in self.structured_inputs:
            input_string += f"{structured_input.name}: {structured_input.leo_type}, "
        return input_string[:-2]

    def generate_input(self, fixed_point_features):
        inputs_without_value = len(
            [input for input in self.input_list if input.value is None]
        )
        if len(fixed_point_features) != inputs_without_value:
            raise Exception(
                f"Number of features ({len(fixed_point_features)}) does not match "
                "number of inputs without a specified value ({inputs_without_value})"
            )

        # Assign values to inputs without a specified value
        index = 0
        for input in self.input_list:
            if input.value is None:
                if isinstance(fixed_point_features, pd.Series):
                    input.tmp_data_value = fixed_point_features.iloc[index]
                else:
                    input.tmp_data_value = fixed_point_features[index]
                index += 1

        formatted_input_list = [
            el.get_input_value_string() for el in self.structured_inputs
        ]
        return formatted_input_list

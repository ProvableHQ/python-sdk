# -*- coding: utf-8 -*-
class _InputGenerator:
    class _Struct:
        MAX_STRUCT_FIELDS = 32
        MAX_STRUCT_HIERARCHY = 32

        def __init__(self, inputs, hierarchy=0):
            self.hierarchy = hierarchy
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
                    self.__class__(chunk, hierarchy=self.hierarchy + 1)
                    for chunk in input_chunks
                ]
            elif (
                num_inputs <= self.MAX_STRUCT_FIELDS
                and hierarchy < self.MAX_STRUCT_HIERARCHY
            ):
                self.fields = inputs
            else:
                raise ValueError("Invalid number of inputs")

    class _Input:
        def __init__(self, value, leo_type, active, name):
            self.value = value
            self.leo_type = leo_type
            self.active = active
            self.name = name
            self.reference_name = None
            self.tmp_data_value = None

        def get_set_value(self):
            if self.value is None:
                return self.tmp_data_value
            else:
                return self.value

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
        self._assign_inputs_to_structs()
        pass

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
            for chunk in input_chunks:
                self.structured_inputs.append(self._Struct(chunk, hierarchy=0))

        a = 0  # Noqa F841

    def generate_input(self, fixed_point_features):
        inputs_without_value = len(
            [input for input in self.input_list if input.value is None]
        )
        if len(fixed_point_features) != inputs_without_value:
            raise Exception(
                f"Number of features ({len(fixed_point_features)}) does not match "
                "number of inputs without a specified value ({inputs_without_value})"
            )

        pass

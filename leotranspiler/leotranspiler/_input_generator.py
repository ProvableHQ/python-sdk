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
            self.reference_name = None
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

    def get_circuit_input_definitions(self):
        return "\n".join(self.struct_definitions)

    def get_circuit_input_string(self):
        self.struct_definitions = set()

        def create_struct(
            inputs, max_fields, depth, max_depth, parent_name="", current_struct_index=0
        ):
            self.struct_definitions = set()
            if depth > max_depth:
                return None, "Exceeded maximum struct hierarchy depth"

            structs = []
            current_struct = ""
            field_count = 0
            struct_count = current_struct_index

            for input in inputs:
                if field_count >= max_fields:
                    structs.append(current_struct[:-2])
                    current_struct = ""
                    field_count = 0
                    struct_count += 1

                reference_name = f"{parent_name}struct{struct_count}.{input.name}"
                input.reference_name = reference_name

                current_struct += f"{input.name}: {input.leo_type}, "
                field_count += 1

            if current_struct:
                structs.append(current_struct[:-2])

            if len(structs) > self.MAX_CIRCUIT_INPUTS:
                nested_structs = []
                nested_index = 0
                for i in range(0, len(structs), max_fields):
                    s, err = create_struct(
                        structs[i : i + max_fields],
                        max_fields,
                        depth + 1,
                        max_depth,
                        f"{parent_name}struct{depth}.",
                        nested_index,
                    )
                    if err:
                        return None, err
                    nested_structs.append(s)
                    nested_index += 1
                return nested_structs, None
            else:
                for i, struct in enumerate(structs):
                    struct_name = f"{parent_name}Struct{i}"
                    self.struct_definitions.add(f"struct {struct_name} {{ {struct} }}")
                return structs, None

        active_inputs = [input for input in self.input_list if input.active]
        active_input_count = len(active_inputs)

        if active_input_count > self.MAX_CIRCUIT_INPUTS:
            structs, err = create_struct(
                active_inputs,
                self.MAX_STRUCT_FIELDS,
                0,
                self.MAX_STRUCT_HIERARCHY,
                "",
                0,
            )
            if err:
                return f"Error: {err}", ""

            struct_strings = []
            for i, _ in enumerate(structs):
                struct_strings.append(f"struct{i}: Struct{i}")

            return ", ".join(struct_strings), self.get_circuit_input_definitions()
        else:
            for input in active_inputs:
                input.reference_name = input.name
            return (
                ", ".join(
                    [f"{input.name}: {input.leo_type}" for input in active_inputs]
                ),
                "",
            )

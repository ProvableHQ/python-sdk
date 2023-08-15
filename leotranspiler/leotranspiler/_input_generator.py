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
    
    def __init__(self, naming_strategy="xi"):
        self.MAX_CIRCUIT_INPUTS = 16
        self.MAX_STRUCT_FIELDS = 32
        self.MAX_STRUCT_HIERARCHY = 32
        
        self.MAX_INPUT_VALUES = self.MAX_CIRCUIT_INPUTS * (self.MAX_STRUCT_FIELDS ** self.MAX_STRUCT_HIERARCHY)
        self.MAX_INPUT_VALUES = 16 # Todo, for now set the value to 16 because hierarchical structs are not implemented yet

        self.inputl_list = []
        self._input_count = 0
        self.naming_strategy = naming_strategy

        if(self.naming_strategy not in ["xi", "custom"]):
            raise Exception("Naming strategy must be one of the following: 'xi', 'custom'")

    def add_input(self, leo_type, active=False, value=None, name=None):
        if(self.naming_strategy == "xi"):
            name = f"x{self._input_count}"
        elif(self.naming_strategy == "custom"):
            if(name == None):
                raise Exception("Custom naming strategy requires a name")

        self.inputl_list.append(self._Input(value, leo_type, active, name))
        self._input_count += 1
    
    def use_input(self, index):
        self.inputl_list[index].active = True
        return self.inputl_list[index]
    
    def get_circuit_input_string(self):
        circuit_inputs_string = ""
        active_input_count = 0
        for input in self.inputl_list:
            if(input.active):
                circuit_inputs_string += f"{input.name}: {input.leo_type}, "
                active_input_count += 1
        
        if(active_input_count == 0):
            raise Exception("No active inputs")
        elif(active_input_count > self.MAX_CIRCUIT_INPUTS):
            raise Exception(f"Too many active inputs ({active_input_count} > {self.MAX_INPUT_VALUES})")
        
        circuit_inputs_string = circuit_inputs_string[:-2]
        return circuit_inputs_string
    
    def generate_input(self, fixed_point_features):
        if(len(fixed_point_features) != len(self.inputl_list)):
            raise Exception(f"Number of features ({len(fixed_point_features)}) does not match number of inputs ({len(self.inputl_list)})")
        
        # assign values to inputs
        for i in range(len(fixed_point_features)):
            self.inputl_list[i].value = fixed_point_features[i]

        # construct input string
        input_string = ""
        for input in self.inputl_list:
            if(input.active):
                input_string += f"{input.value}{input.leo_type} "
        input_string = input_string[:-1]
        
        return input_string
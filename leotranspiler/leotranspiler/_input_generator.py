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
        self.active_inputs_count = 0

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
        for input in self.inputl_list:
            if(input.active):
                circuit_inputs_string += f"{input.name}: {input.leo_type}, "
        circuit_inputs_string = circuit_inputs_string[:-2]
        return circuit_inputs_string
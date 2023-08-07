class _InputGenerator:
    class _Struct:
        def __init__(self, fields=None, hierarchy=0):
            self.fields = fields
            self.hierarchy = hierarchy
    
    def __init__(self):
        self.MAX_CIRCUIT_INPUTS = 16
        self.MAX_STRUCT_FIELDS = 32
        self.MAX_STRUCT_HIERARCHY = 32
        
        self.max_input_values = self.MAX_CIRCUIT_INPUTS * (self.MAX_STRUCT_FIELDS ** self.MAX_STRUCT_HIERARCHY)
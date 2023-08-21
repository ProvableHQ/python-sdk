class _ComputationBase:
    def __init__(self, input, outputs_original, circuit_constraints, outputs_decimal, fixed_point_scaling_factor=1):
        self.input = input
        self.outputs_original = outputs_original
        self.circuit_constraints = circuit_constraints
        self.outputs_decimal = outputs_decimal
        self.fixed_point_scaling_factor = fixed_point_scaling_factor

class ZeroKnowledgeProof(_ComputationBase):
    def __init__(self, input, outputs_original, circuit_constraints, proof, outputs_decimal=None):
        super().__init__(input, outputs_original, circuit_constraints, outputs_decimal)
        self.proof = proof

class LeoComputation(_ComputationBase):
    def __init__(self, input, outputs_original, circuit_constraints, outputs_decimal=None):
        super().__init__(input, outputs_original, circuit_constraints, outputs_decimal)
class _ComputationBase:
    def __init__(self, input, output, circuit_constraints, output_decimal, fixed_point_scaling_factor=1):
        self.input = input
        self.output = output
        self.circuit_constraints = circuit_constraints
        self.output_decimal = output_decimal
        self.fixed_point_scaling_factor = fixed_point_scaling_factor

class ZeroKnowledgeProof(_ComputationBase):
    def __init__(self, input, output, circuit_constraints, proof, execution=None, output_decimal=None):
        super().__init__(input, output, circuit_constraints, output_decimal)
        self.proof = proof
        self.execution = execution

class LeoComputation(_ComputationBase):
    def __init__(self, input, output, circuit_constraints, output_decimal=None):
        super().__init__(input, output, circuit_constraints, output_decimal)
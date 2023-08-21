class _ComputationBase:
    def __init__(self, input, output, circuit_constraints):
        self.input = input
        self.output = output
        self.circuit_constraints = circuit_constraints

class ZeroKnowledgeProof(_ComputationBase):
    def __init__(self, input, output, circuit_constraints, proof):
        super().__init__(input, output, circuit_constraints)
        self.proof = proof

class LeoComputation(_ComputationBase):
    def __init__(self, input, output, circuit_constraints):
        super().__init__(input, output, circuit_constraints)
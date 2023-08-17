class ZeroKnowledgeProof:
    def __init__(self, input, output, proof, circuit_constraints=None):
        self.input = input
        self.output = output
        self.proof = proof
        self.circuit_constraints = circuit_constraints
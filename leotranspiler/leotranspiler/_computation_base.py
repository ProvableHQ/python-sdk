# -*- coding: utf-8 -*-
class _ComputationBase:
    """
    A base class for computational structures.

    Attributes
    ----------
    - input: The input for the computation.
    - output: The expected output after computation.
    - circuit_constraints: Constraints for the computation.
    - output_decimal: Decimal representation of the output.
    - fixed_point_scaling_factor: A scaling factor (defaults to 1).
    """

    def __init__(
        self,
        input,
        output,
        circuit_constraints,
        output_decimal,
        fixed_point_scaling_factor=1,
    ):
        self.input = input
        self.output = output
        self.circuit_constraints = circuit_constraints
        self.output_decimal = output_decimal
        self.fixed_point_scaling_factor = fixed_point_scaling_factor


class ZeroKnowledgeProof(_ComputationBase):
    """
    Represents a Zero Knowledge Proof based on the ComputationBase structure.

    Attributes
    ----------
    - proof: The zero-knowledge proof.
    - execution: (Optional) Details of the execution of the proof.

    Inherits other attributes from _ComputationBase.
    """

    def __init__(
        self,
        input,
        output,
        circuit_constraints,
        proof,
        execution=None,
        output_decimal=None,
    ):
        super().__init__(input, output, circuit_constraints, output_decimal)
        self.proof = proof
        self.execution = execution


class LeoComputation(_ComputationBase):
    """
    Represents a Leo Computation structure based on the ComputationBase.

    Inherits attributes from _ComputationBase.
    """

    def __init__(self, input, output, circuit_constraints, output_decimal=None):
        super().__init__(input, output, circuit_constraints, output_decimal)

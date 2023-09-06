# -*- coding: utf-8 -*-
class _ComputationBase:
    """
    A base class for computational structures.

    Attributes
    ----------
    - input: The input for the computation.
    - output: The output after computation.
    - circuit_constraints: Constraints for the computation.
    - active_input_count: The number of active inputs.
    - output_decimal: Decimal representation of the output.
    - runtime: The runtime of the computation in seconds.
    - fixed_point_scaling_factor: A scaling factor (defaults to 1).
    """

    def __init__(
        self,
        input,
        output,
        circuit_constraints,
        active_input_count,
        output_decimal,
        runtime,
        fixed_point_scaling_factor=1,
    ):
        self.input = input
        self.output = output
        self.circuit_constraints = circuit_constraints
        self.active_input_count = active_input_count
        self.output_decimal = output_decimal
        self.runtime = runtime
        self.fixed_point_scaling_factor = fixed_point_scaling_factor


class ZeroKnowledgeProof(_ComputationBase):
    """
    Represents a Zero Knowledge Proof based on the ComputationBase structure.

    Attributes
    ----------
    - input: The input for the computation.
    - output: The output after computation.
    - circuit_constraints: Constraints for the computation.
    - active_input_count: The number of active inputs.
    - proof: The zero-knowledge proof.
    - execution: (Optional) Details of the execution of the proof.
    - output_decimal: Decimal representation of the output.

    Inherits other attributes from _ComputationBase.
    """

    def __init__(
        self,
        input,
        output,
        circuit_constraints,
        active_input_count,
        runtime,
        proof,
        execution=None,
        output_decimal=None,
    ):
        super().__init__(
            input,
            output,
            circuit_constraints,
            active_input_count,
            output_decimal,
            runtime,
        )
        self.proof = proof
        self.execution = execution


class LeoComputation(_ComputationBase):
    """
    Represents a Leo Computation structure based on the ComputationBase.

    Inherits attributes from _ComputationBase.
    """

    def __init__(
        self,
        input,
        output,
        circuit_constraints,
        active_input_count,
        runtime,
        output_decimal=None,
    ):
        super().__init__(
            input,
            output,
            circuit_constraints,
            active_input_count,
            output_decimal,
            runtime,
        )

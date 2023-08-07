from .zero_knowledge_proof import ZeroKnowledgeProof
class LeoTranspiler:
    def __init__(self, model, validation_data=None, model_as_input=False, ouput_model_hash=None):
        """
        Create a new transpiler instance.

        Parameters
        ----------
        model : Model
            The model to transpile.
        validation_data : tuple of array_like, optional
            Data on which to evaluate the numerical stability of the circuit. The model will not be trained on
            this data.
        model_as_input : bool, optional
            If True, the model weights and biases are treated as circuit input instead of being hardcoded.
        ouput_model_hash : str, optional
            If set, the circuit will return the hash of the model weights and biases. Possible values are ... (todo)

        Returns
        -------
        LeoTranspiler
            The transpiler instance.
        """

        self.model = model
        self.validation_data = validation_data
        self.model_as_input = model_as_input
        self.ouput_model_hash = ouput_model_hash

        self.transpilation_result = None
        self.leo_program_stored = False

    def _transpile(self):
        """
        Transpile the model to Leo.

        Returns
        -------
        None
        """
        self.transpilation_result = "TODO"

    def store_leo_program(self, path):
        """
        Store the Leo program to a file.

        Parameters
        ----------
        path : str
            The path to the file to store the Leo program in.

        Returns
        -------
        None
        """ 
        if self.transpilation_result is None:
            print("Transpiling model...")
            self._transpile()
        with open(path, "w") as f:
            f.write(self.transpilation_result)
        self.leo_program_stored = True
        print("Leo program stored")

    def prove(self, input):
        """
        Prove the model output for a given input.

        Parameters
        ----------
        input : array_like
            The input for which to prove the output.

        Returns
        -------
        ZeroKnowledgeProof
            The zero-knowledge proof for the given input.
        """
        if not self.leo_program_stored:
            raise Exception("Leo program not stored")
        
        circuit_input = None # TODO: organize circuit input
        circuit_output, proof_value = None, None # TODO: here we need to do the FFI call or CLI call for leo/snarkVM execute
        return ZeroKnowledgeProof(circuit_input, circuit_output, proof_value)
    
    def decision_tree_to_pseudocode(self, tree, feature_names, node=0, indentation=""):
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        # Base case: leaf node
        if left_child == right_child:  # means it's a leaf
            return indentation + f"Return {tree.value[node].argmax()}\n"

        # Recursive case: internal node
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        if node == 0:
            pseudocode = f"IF {feature} <= {threshold:.2f} THEN\n"
        else:
            pseudocode = indentation + f"IF {feature} <= {threshold:.2f} THEN\n"
        
        pseudocode += self.decision_tree_to_pseudocode(tree, feature_names, left_child, indentation + "    ")
        pseudocode += indentation + "ELSE\n"
        pseudocode += self.decision_tree_to_pseudocode(tree, feature_names, right_child, indentation + "    ")
        return pseudocode
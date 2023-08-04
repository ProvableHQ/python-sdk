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
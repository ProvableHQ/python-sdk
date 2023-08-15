from .zero_knowledge_proof import ZeroKnowledgeProof
from ._model_transpiler import _get_model_transpiler
import os, time, subprocess, psutil

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

    def store_leo_program(self, path, project_name):
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

        self.model_transpiler = _get_model_transpiler(self.model, self.validation_data)

        # Check numeric stability for model and data and get number range
        self.model_transpiler._numbers_get_leo_type_and_fixed_point_scaling_factor()

        if self.transpilation_result is None:
            print("Transpiling model...")
            self.transpilation_result = self.model_transpiler.transpile(project_name) # todo check case when project name changes

        project_dir = os.path.join(path, project_name)
        src_folder_dir = os.path.join(project_dir, "src")
        leo_file_dir = os.path.join(src_folder_dir, "main.leo")

        # Make sure path exists
        os.makedirs(src_folder_dir, exist_ok=True)

        with open(leo_file_dir, "w") as f:
            f.write(self.transpilation_result)

        program_json = self._get_program_json(project_name)
        program_json_file_dir = os.path.join(project_dir, "program.json")
        with open(program_json_file_dir, "w") as f:
            f.write(program_json)
        
        environment_file = self._get_environment_file() # todo option to pass private key
        environment_file_dir = os.path.join(project_dir, ".env")
        with open(environment_file_dir, "w") as f:
            f.write(environment_file)

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
        
        circuit_input = self.model_transpiler.generate_input(input)

        # TODO: here we need to do the FFI call or CLI call for leo/snarkVM execute
        circuit_output, proof_value = None, None
        command = ['leo', 'run', 'main'] + circuit_input
        directory = os.path.join(os.getcwd(), "leotranspiler", "tests", "tree1")

        # Start Leo program
        start = time.time()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory)

        while process.poll() is None:
            try:
                time.sleep(0.1)
            except psutil.NoSuchProcess:
                break

        end = time.time()

        # Get the output
        stdout, stderr = process.communicate()
        result = stdout.decode() + stderr.decode()
        runtime = end - start

        # Check if "Finished" is in the results string
        success = "Finished" in result
        if success:
            # Extract the number before the word "constraints" in the results string
            constraints = result.split("constraints")[0].split()[-1].replace(",", "")
            constraints = int(constraints)
            # Todo output processing
        else:
            print("Error:", result)

        return ZeroKnowledgeProof(circuit_input, circuit_output, proof_value)
    
    def _get_program_json(self, project_name):
        """
        Generate the program.json file content.

        Parameters
        ----------
        project_name : str
            The name of the project.

        Returns
        -------
        str
            The program.json file.
        """
        return f"""{{
    "program": "{project_name}.aleo",
    "version": "0.0.0",
    "description": "transpiler generated program",
    "license": "MIT"
}}"""

    def _get_environment_file(self):
        """
        Generate the environment file content.

        Returns
        -------
        str
            The environment file.
        """
        return f"""NETWORK=testnet3
PRIVATE_KEY=APrivateKey1zkpHtqVWT6fSHgUMNxsuVf7eaR6id2cj7TieKY1Z8CP5rCD
"""
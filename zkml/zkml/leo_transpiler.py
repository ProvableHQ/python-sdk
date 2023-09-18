# -*- coding: utf-8 -*-
"""Provides the LeoTranspiler class for transpiling ML models.

This module enables transpilation of machine learning models into the Leo
programming language, suitable for creating zero-knowledge proofs.
"""
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import psutil
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from ._computation_base import LeoComputation, ZeroKnowledgeProof
from ._model_transpiler import _get_model_transpiler


class LeoTranspiler:
    """Main class for transpiling machine learning models into the Leo language.

    LeoTranspiler takes a machine learning model, and potentially some validation data,
    and provides methods to transpile this model into a Leo program. This transpiled
    program can be run to generate zero-knowledge proofs for given input samples.
    """

    def __init__(
        self,
        model: BaseEstimator,
        validation_data: Optional[ArrayLike] = None,
        output_model_hash: Optional[str] = None,
    ):
        """Initialize the LeoTranspiler with the given parameters.

        Parameters
        ----------
        model : BaseEstimator
            The ML model to transpile.
        validation_data : tuple of array_like, optional
            Data to evaluate the numerical stability of the circuit. The model will
            not be trained on this data.
        output_model_hash : str, optional
            If provided, the circuit returns the hash of the model's weights and
            biases. This functionality is not yet implemented.
        """
        self.model = model
        self.validation_data = validation_data
        self.output_model_hash = output_model_hash

        self.model_as_input = None
        self.transpilation_result = None
        self.leo_program_stored = False

    def to_leo(self, path: Path, project_name: str, model_as_input: bool = False):
        """Transpile and store the Leo program to a specified directory.

        This method transpiles the model to a Leo program and saves it, along with
        related configuration files, to the specified path under a directory named
        after the project name.

        Parameters
        ----------
        path : Path
            The directory where the Leo program and related files will be stored.
        project_name : str
            The name of the project, which determines the directory name under
            the specified path and is used during transpilation.
        model_as_input : bool, optional
            If True, the model's weights and biases are treated as circuit input
            rather than being hardcoded.

        Returns
        -------
        None

        Notes
        -----
        The method saves three main files:
        - `main.leo`: The transpiled Leo program.
        - `program.json`: Configuration for the Leo program.
        - `.env`: Environment-specific settings.

        If the Leo program is already transpiled, this method will not re-transpile
        but will directly store the program.

        Directories are created as needed to ensure the specified path exists.
        """
        self._check_installed_leo_version()
        self.model_transpiler = _get_model_transpiler(self.model, self.validation_data)

        if self.transpilation_result is None or self.model_as_input != model_as_input:
            # Computing the number ranges and the fixed-point scaling factor
            logging.info("Computing number ranges and fixed-point scaling factor...")
            self.model_transpiler._numbers_get_leo_type_and_fixed_point_scaling_factor()
            logging.info("Transpiling model...")
            self.transpilation_result = self.model_transpiler.transpile(
                project_name,
                model_as_input=model_as_input,
            )  # todo check case when project name changes
            self.model_as_input = model_as_input

        self.project_dir = os.path.join(path, project_name)
        self._store_leo_program()
        self._store_program_json(project_name)
        self._store_environment_file()  # todo implement option to pass private key

        self.leo_program_stored = True
        logging.info("Leo program stored")

    def run(self, input_sample: Union[ndarray, List[float]]) -> LeoComputation:
        """Run the model in Leo output for a given input sample.

        Parameters
        ----------
        input_sample : Union[ndarray, List[float]]
            The input sample for which to prove the output. Can be a numpy array or a
            list of floats.

        Returns
        -------
        LeoComputation
            The Leo computation for the given input sample.
        """
        leo_computation = self._handle_input_sample(input_sample, "run")

        return leo_computation

    def execute(self, input_sample: Union[ndarray, List[float]]) -> ZeroKnowledgeProof:
        """Run the model in Leo output for a given input sample.

        Parameters
        ----------
        input_sample : Union[ndarray, List[float]]
            The input sample for which to prove the output. Can be a numpy array or a
            list of floats.

        Returns
        -------
        ZeroKnowledgeProof
            The zero knowledge proof for the given input sample.
        """
        zkp = self._handle_input_sample(input_sample, "execute")

        return zkp

    def _handle_input_sample(self, input_sample, command):
        if not self.leo_program_stored:
            raise FileNotFoundError("Leo program not stored")

        computation_base_result = []

        if isinstance(input_sample, pd.DataFrame):  # an entire dataset
            for _, row in input_sample.iterrows():
                computation_base_result.append(self._handle_run_execute(row, command))
        elif isinstance(input_sample, list):  # a list of data points
            for data_point in input_sample:
                computation_base_result.append(
                    self._handle_run_execute(data_point, command)
                )
        elif (
            isinstance(input_sample, ndarray)
            and self.validation_data is not None
            and input_sample.ndim == self.validation_data.ndim
        ):
            for data_point in input_sample:
                computation_base_result.append(
                    self._handle_run_execute(data_point, command)
                )
        elif isinstance(input_sample, ndarray) and self.validation_data is None:
            logging.warning(
                "No validation_data passed to the transpiler, thus, no information "
                "available of dataset shape. "
                f"Passed input sample for {command} is treated as a single data point"
            )
            computation_base_result = self._handle_run_execute(input_sample, command)
        else:  # a single data point
            computation_base_result = self._handle_run_execute(input_sample, command)

        return computation_base_result

    def _handle_run_execute(self, input_sample, command):
        circuit_inputs_fixed_point = self.model_transpiler.generate_input(input_sample)
        result, runtime = self._execute_leo_cli(command, circuit_inputs_fixed_point)
        computation_base_result = self._parse_leo_output(
            command, result, circuit_inputs_fixed_point, runtime
        )
        self.model_transpiler.convert_computation_base_outputs_to_decimal(
            computation_base_result
        )

        return computation_base_result

    def _execute_leo_cli(self, command: str, inputs: List[str]) -> Tuple[str, float]:
        """Execute a Leo CLI command.

        Parameters
        ----------
        command : str
            The command to execute.
        inputs : List[str]
            The inputs to the command.

        Returns
        -------
        Tuple[str, float]
            The command output and the runtime in seconds.
        """
        directory = self.project_dir

        # Start Leo program
        start = time.time()
        process = subprocess.Popen(
            ["leo", command, "main"] + inputs,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
        )

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

        return result, runtime

    def _parse_leo_output(
        self,
        command: str,
        result: str,
        input: Optional[Union[ndarray, List[float]]] = None,
        runtime: Optional[float] = None,
    ) -> Union[LeoComputation, ZeroKnowledgeProof]:
        """Parse the Leo output.

        Parameters
        ----------
        command : str
            The command that was executed.
        result : str
            The Leo output.
        input : Union[ndarray, List[float]], optional
            The input sample for which the output was computed, by default None
        runtime : float, optional
            The runtime of the Leo computation in seconds, by default None

        Returns
        -------
        Union[LeoComputation, ZeroKnowledgeProof]
            - If the command was "run", a `LeoComputation` object.
            - If the command was "execute", a `ZeroKnowledgeProof` object.

        Raises
        ------
        ValueError
            If the command is not recognized or if there was an error parsing the
            output.
        """
        outputs_fixed_point = []

        success_run = "Finished" in result and command == "run"
        success_execute = "Executed" in result and command == "execute"
        if success_run or success_execute:
            constraints = int(
                result.split("constraints")[0].split()[-1].replace(",", "")
            )
            # Output processing
            outputs_str = result.split("Output")[1]
            outputs_str = outputs_str.split("• ")
            for element in outputs_str:
                if element.startswith("\n"):
                    continue
                # check if is number
                if element[0].isdigit():
                    element = element.split(self.model_transpiler.leo_type)[0]
                    outputs_fixed_point.append(int(element))
        else:
            logging.error(f"Error while parsing leo outputs: {result}")
            raise ValueError("Error while parsing leo outputs")

        if success_execute:
            # get index of last \n\n in result
            index = result.rfind("\n\n")
            result_content = result[:index]
            index = result_content.rfind("\n\n")
            result_content = result_content[index + 2 :]
            # parse json
            execution_data = json.loads(result_content)

        if command == "run":
            return LeoComputation(
                input,
                outputs_fixed_point,
                constraints,
                self.model_transpiler.active_input_count,
                runtime,
            )
        elif command == "execute":
            return ZeroKnowledgeProof(
                input,
                outputs_fixed_point,
                constraints,
                self.model_transpiler.active_input_count,
                runtime,
                execution_data["execution"]["proof"],
                execution_data["execution"],
            )
        else:
            raise ValueError("Unknown command")

    def _store_leo_program(self):
        """Store the Leo program.

        Parameters
        ----------
        project_name : str
            The name of the project.

        Returns
        -------
        None
        """
        folder_dir = os.path.join(self.project_dir, "src")
        # Make sure path exists
        os.makedirs(folder_dir, exist_ok=True)

        with open(os.path.join(folder_dir, "main.leo"), "w") as f:
            f.write(self.transpilation_result)

    def _store_program_json(self, project_name: str):
        """Store the program.json file.

        Parameters
        ----------
        project_name : str
            The name of the project.

        Returns
        -------
        None
        """
        content = f"""{{
    "program": "{project_name}.aleo",
    "version": "0.0.0",
    "description": "transpiler generated program",
    "license": "MIT"
}}"""
        folder_dir = os.path.join(self.project_dir)
        # Make sure path exists
        os.makedirs(folder_dir, exist_ok=True)

        with open(os.path.join(folder_dir, "program.json"), "w") as f:
            f.write(content)

    def _store_environment_file(self):
        """Store the environment configuration file.

        Returns
        -------
        None
        """
        content = """NETWORK=testnet3
PRIVATE_KEY=APrivateKey1zkpHtqVWT6fSHgUMNxsuVf7eaR6id2cj7TieKY1Z8CP5rCD
"""
        folder_dir = os.path.join(self.project_dir)
        # Make sure path exists
        os.makedirs(folder_dir, exist_ok=True)

        with open(os.path.join(folder_dir, ".env"), "w") as f:
            f.write(content)

    def _check_installed_leo_version(self):
        """Check if Leo is installed and the version is up to date.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the Leo version is not correct.
        FileNotFoundError
            If Leo is not installed.
        """
        MIN_LEO_VERSION = "1.9.3"

        def parse_version(version):
            return tuple(map(int, version.split(".")))

        try:
            version = (
                subprocess.check_output(["leo", "--version"])
                .decode()
                .strip()
                .split(" ")[1]
            )
            if parse_version(version) < parse_version(MIN_LEO_VERSION):
                raise ValueError(
                    f"Leo version must be at least {MIN_LEO_VERSION}. Please update "
                    f"Leo using `leo update`."
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Leo not installed. Please visit "
                "https://developer.aleo.org/leo/installation/ to install Leo."
            )

# Python to Leo - machine learning model transpiler

This Python library provides tools to transpile Python machine learning models into Leo code. It also provides tools to run the transpiled code from python and to create zk proofs.

## Guide for users (i.e., Python ML developers)

1. Ensure you have the Python 3 of version 3.9.6 or newer installed: You can check it by running "python3 --version" Here is the instructions to install Python ...
2. Ensure you have Leo version 1.9.3 or newer installed.  https://developer.aleo.org/leo/installation/ You can check it by running "leo --version"
1. Ensure you are on the master branch of GitHub (by default you should be)
3. Clone the repository, or download a .whl or .tar.gz from the dist folder
4. cd to the folder containing the .whl/.tar.gz file
5. Install it via either "pip3 install leotranspiler-0.1.0-py3-none-any.whl" or "pip3 install leotranspiler-0.1.0.tar.gz"
6. Use it. In the examples folder, you can find example usage

Notes:
- On some machines, you may need to run "python" and "pip" instead of "python3" and "pip3"

## Guide for library developers

1. Clone the repository
2. Make sure you have not already installed a version of leotranspiler, e.g. by running pip3 uninstall leotranspiler
3. cd into the src directory leotranspiler and run pip3 install -e .
4. Code, and advance the library
5. Run the tests
6. If you wish to create executables, make sure you have poetry installed. Then, run "poetry build"
7. If you with to commit, please ensure you are working on your own branch, and ensure the pre-commit tests pass. You can check this by running pre-commit run --all-files. For this, you need to have pre-comimt installed: ...

Notes:
- When coding, please follow the PEP 8 style guide: https://peps.python.org/pep-0008/, and make sure you write docstrings for public-facing functions and classes
- More detailed contributing instructions will follow. In the meantime, if you wish to contribute, please reach out to us

# Python to Leo: Machine Learning Model Transpiler

This Python library offers the ability to transpile Python machine learning models into Leo code. Additionally, it provides tools to execute the transpiled code from Python and to generate zk proofs.

## 🚀 Getting Started for Users (Python ML Developers)

### Prerequisites:

1. **Python**: Ensure Python 3.9.6 or newer is installed.
   - Check by running:
   ```bash
   python3 --version
   ```
   - If not installed, follow the instructions [here](https://wiki.python.org/moin/BeginnersGuide/Download).

2. **Leo**: Ensure Leo version 1.9.3 or newer is installed.
   - Check by running:
   ```bash
   leo --version
   ```
   - If necessary, update:
   ```bash
   leo update
   ```
   - Installation guide: [Leo Installation](https://developer.aleo.org/leo/installation/)

3. Confirm you're on the master branch of GitHub (you should be by default).

### Installation:

1. Clone the repository, or download the `.whl` or `.tar.gz` file from the `dist` folder.
2. Navigate to the directory containing the `.whl/.tar.gz` file:
   ```bash
   cd PATH_TO_DIRECTORY
   ```
3. Install using pip:
   ```bash
   pip3 install leotranspiler-0.1.0-py3-none-any.whl
   ```
   OR
   ```bash
   pip3 install leotranspiler-0.1.0.tar.gz
   ```

### Usage:

- Explore the `examples` folder for example usages.

**Notes**:
- On some systems, "python" and "pip" might be used instead of "python3" and "pip3".

## 🛠 Guide for Library Developers

### Setup:

1. Clone the repository.
2. Ensure no previous version of `leotranspiler` is installed:
   ```bash
   pip3 uninstall leotranspiler
   ```
3. Navigate to the `leotranspiler` source code directory:
   ```bash
   cd leotranspiler
   ```
4. Install in editable mode:
   ```bash
   pip3 install -e .
   ```

### Development:

1. Code and enhance the library.
2. Ensure you run the tests.
3. If you want to generate executables, ensure `poetry` is installed and then run:
   ```bash
   poetry build
   ```

### Committing:

1. Ensure you're working on your own branch.
2. Make sure the pre-commit tests pass. Check by running:
   ```bash
   pre-commit run --all-files
   ```
   - Install `pre-commit` if not present [installation link here].

**Notes**:
- Adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/) and ensure you provide docstrings for all public-facing functions and classes.
- More detailed contribution guidelines will be provided soon. If you'd like to contribute in the meantime, please contact us.

---

Thank you for your interest in Python to Leo transpiler. Let's push the boundaries of zk and Python together!

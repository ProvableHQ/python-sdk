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

**Through PyPI**:

- Install using the following command:
   ```bash
   pip3 install zkml
   ```

**Through the GitHub repository**:

1. Clone the repository, or download the `.whl` or `.tar.gz` file from the `dist` folder.
2. Navigate to the `dist` directory containing the `.whl` or `.tar.gz` file:
   ```bash
   cd dist
   ```
3. Install using pip:
   ```bash
   pip3 install zkml-0.0.1b1-py3-none-any.whl
   ```
   OR
   ```bash
   pip3 install zkml-0.0.1b1.tar.gz
   ```

### Usage:

- Explore the `examples` folder from GitHub for example usages. To run the examples, additional Python packages are required. You can install these from the `examples` folder by running:
   ```bash
   pip3 install -r requirements.txt
   ```
- The examples demonstrate how to work with the Python to Leo transpiler. Currently, the transpiler supports sklearn decision tree models, and the examples cover the Iris dataset, the German credit dataset, and the MNIST dataset.

**Notes**:
- On some systems, "python" and "pip" might be used instead of "python3" and "pip3".
- In case you are unfamiliar with Jupyter notebooks, here are two ways to run these notebooks:

  1. **Visual Studio Code (VS Code)**
      - Ensure you have [VS Code installed](https://code.visualstudio.com/).
      - [Install the Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in VS Code.
      - Open the notebook file (`.ipynb`) in VS Code to view, edit, and run the Python code cells interactively.

  2. **Jupyter Notebook**
      - Ensure you have [Jupyter Notebook installed](https://jupyter.org/install.html).
      - Navigate to the `examples` folder in a terminal and launch Jupyter Notebook using the command `jupyter notebook`.
      - Once Jupyter Notebook launches in your browser, open the notebook files (`.ipynb`) to view and run the Python code cells interactively.

  For a more detailed tutorial on using Jupyter Notebooks, refer to this [Jupyter Notebook beginner guide](https://realpython.com/jupyter-notebook-introduction/).


## 🛠 Guide for Library Developers

### Setup:

1. Clone the repository.
2. Ensure no previous version of `zkml` is installed:
   ```bash
   pip3 uninstall zkml
   ```
3. Navigate to the `zkml` source code directory:
   ```bash
   cd zkml
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
   - Install `pre-commit` if not present ([installation link here](https://pre-commit.com/)).

**Notes**:
- Adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/) and ensure you provide docstrings for all public-facing functions and classes.
- More detailed contribution guidelines will be provided soon. If you'd like to contribute in the meantime, please contact us via email.

---

Thank you for your interest in the zkml Python to Leo transpiler. Let's push the boundaries of zk and Python together!

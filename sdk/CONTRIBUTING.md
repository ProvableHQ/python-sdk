# Contributing

Thank you for your interest in contributing to the Aleo python-sdk! The Aleo python-sdk consists of multiple libraries, including the Aleo python library, and the zkml library. Below you can find some guidelines that the projects strive to follow.

## Filing Issues

When filing a new issue:

- Provide a clear and concise title.
- Describe the issue in detail, including steps to reproduce, expected behavior, and observed behavior.
- Label the issue correctly, e.g., "bug", "enhancement", etc.

## Development process

1. Code and enhance the respective library with new/better functionalities.
2. Write test functions to test the new functionalities.
3. Ensure all of the tests run successfully.
4. If you want to generate executables, ensure `poetry` is installed and then run:
   ```bash
   poetry build
   ```

## Pull requests

Please follow the instructions below when filing pull requests:

- Ensure that your branch is forked from the current [master](https://github.com/AleoHQ/python-sdk/tree/master) branch.
- Provide descriptive text for the feature or proposal. Be sure to link the pull request to any issues by using keywords. Example: "closes #130".
- Write clear and concise commit messages, describing the changes you made.
- For the zkml library only: Run `pre-commit run --all-files` before you commit. You can find the installation details for pre-commit [here](https://pre-commit.com/). The pre-commit hooks, as configured in this repository, enforce PEP 8 ([PEP 8 style guide](https://peps.python.org/pep-0008/)), including docstrings for all public-facing functions and classes. For details, please refer to the `.pre-commit-config.yaml` file.

# Contributing

Thank you for your interest in contributing to the zkml Leo transpiler! Below you can find some guidelines that the project strives to follow.

## Development process:

1. Code and enhance the library with new/better functionalities.
2. Write test functions to test the new functionalities.
3. Ensure all of the tests run successfully.
4. If you want to generate executables, ensure `poetry` is installed and then run:
   ```bash
   poetry build
   ```

## Pull requests

Please follow the instructions below when filing pull requests:

- Ensure that your branch is forked from the current [master](https://github.com/AleoHQ/python-sdk/zkml/tree/master) branch.
- Fill out the provided markdown template for the feature or proposal. Be sure to link the pull request to any issues by using keywords. Example: "closes #130".
- Run `pre-commit run --all-files` before you commit. You can find the installation details for pre-commit [here](https://pre-commit.com/). The pre-commit checks include the [PEP 8 style guide](https://peps.python.org/pep-0008/) and enforcing docstrings for all public-facing functions and classes

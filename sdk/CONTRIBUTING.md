# Contributing

Thank you for your interest in contributing to the Aleo sdk library of the Aleo python-sdk! Below you can find some guidelines that the projects strive to follow.

## Filing Issues

When filing a new issue:

- Provide a clear and concise title.
- Describe the issue in detail, including steps to reproduce, expected behavior, and observed behavior.
- Label the issue correctly, e.g., "bug", "enhancement", etc.

## Development process

1. Code and enhance the respective library with new/better functionalities.
2. Write test functions to test the new functionalities into the `test.py` file, using the `unittest` library.
3. Ensure all of the tests run successfully, using the `install.sh` file.
4. Please also ensure the CircleCI tests run successfully, as these tests will be enforced on GitHub prior to merging a pull request. For this, ensure you have `circleci-cli` [installed](https://circleci.com/docs/local-cli/), and `Docker` [installed](https://docs.docker.com/engine/install/). Then, navigate to the `python-sdk` root folder of the repository and run:
   ```bash
   circleci local execute build-and-test
   ```
5. If you want to generate executables for the release, ensure `poetry` is installed and then run:
   ```bash
   poetry build
   ```
   In the future, we aim to automate this step with automated building through CircleCI.

## Pull requests

Please follow the instructions below when filing pull requests:

- Ensure that your branch is forked from the current [master](https://github.com/AleoHQ/python-sdk/tree/master) branch.
- Provide descriptive text for the feature or proposal. Be sure to link the pull request to any issues by using keywords. Example: "closes #130".
- Write clear and concise commit messages, describing the changes you made.

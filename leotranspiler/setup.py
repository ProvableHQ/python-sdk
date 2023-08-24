# -*- coding: utf-8 -*-
"""Setup script for the LeoTranspiler package."""
from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="leotranspiler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["psutil"],
    author="Konstantin Pandl, Michael Turner",
    author_email="konstantin@aleo.org, mturner@aleo.org",
    description="A transpiler for Python machine learning models to Leo for "
    "zero-knowledge inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AleoHQ/python-sdk/tree/master/leotranspiler",
)

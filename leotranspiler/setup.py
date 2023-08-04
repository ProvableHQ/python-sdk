from setuptools import setup, find_packages

setup(
    name='leotranspiler',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list of dependencies
    ],
    author='Konstantin Pandl, Michael Turner',
    author_email='konstantin@aleo.org, mturner@aleo.org',
    description='A transpiler for Python machine learning models to Leo for zero-knowledge inference',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AleoHQ/python-sdk/tree/master/leotranspiler',
)
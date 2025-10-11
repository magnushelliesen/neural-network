from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='NeuralNetwork',
    version='0.1.6',
    url='https://github.com/magnushelliesen/neural-network.git',
    author='Magnus Kvåle Helliesen',
    author_email='magnus.helliesen@gmail.com',
    description='',
    packages=find_packages(),    
    install_requires=required,
)
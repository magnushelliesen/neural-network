from setuptools import setup, find_packages

setup(
    name='NeuralNetwork',
    version='0.1.4',
    url='https://github.com/magnushelliesen/neural-network.git',
    author='Magnus Kvåle Helliesen',
    author_email='magnus.helliesen@gmail.com',
    description='',
    packages=find_packages(),    
    install_requires=['numpy == 2.0.0', 'pandas == 2.2.2', 'matplotlib == 3.9.0'],
)
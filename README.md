# neural-network
I've built a [neural network-class](https://github.com/magnushelliesen/neural-network/blob/main/neural_network/neural_network.py) (from first principles, i.e., calculus and linear algebra only, using NumPy) with a train/backpropagation method and everything.

The class can be installed using
```python
pip install git+https://github.com/magnushelliesen/neural-network.git
```

The class seems to be able to train on the [MNIST dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) (see [notebook](https://github.com/magnushelliesen/neural-network/blob/main/notebooks/demo-neural-network-mnist-test.ipynb)), it recognizes most (but not all) handwritten digits. Look how effortlessly it recognizes even this squiggly "9":

![image](https://github.com/magnushelliesen/neural-network/assets/104299371/11f036eb-f39b-4ffb-b413-398532a93f72)

It was a lot of tinkering to get it flying, lemme tell ya! What did the trick was to switch from batch gradient descent to stochastic gradient descent (and removing som bugs/errors). This is really cool!

I'm considering renaming this project/class "SonOfAnton" (for those who get the reference).

![image](https://github.com/magnushelliesen/neural-network/assets/104299371/3aa8d57c-5754-4687-9aef-9b5b6b603ab7)

## Testing
This repo is set up with two GitHub Actions to make sure I don't merge code that breaks stuff into main:
1. A [GitHub Action](https://github.com/magnushelliesen/neural-network/blob/main/.github/workflows/run-tests.yml) that runs unit tests on the class upon PR into main
2. A [GitHub Action](https://github.com/magnushelliesen/neural-network/blob/main/.github/workflows/pre-commit-checks.yml) that runs Mypy and Black upon PR into main

## Neural network app
A pre-trained instance of the class is hosted on [Google Cloud Run](https://handwritten-digit-recognizer-app-569320861368.asia-east1.run.app/) and provides a nifty user interface.

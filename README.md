# neural-network
I've built a neural network-[class](https://github.com/magnushelliesen/neural-network/blob/main/neural_network/neural_network.py) (from first principles, only using NumPy) with a train/backpropagation method. 

The class seems to be able to train on the [MNIST dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) (see [notebook](https://github.com/magnushelliesen/neural-network/blob/main/neural-network-mnist-test.ipynb)), it recognizes most (but not all) digits. There's probably something left to work on with the backpropagation method. But DANG(!), it can read a squiggly 9:

![image](https://github.com/magnushelliesen/neural-network/assets/104299371/11f036eb-f39b-4ffb-b413-398532a93f72)

It was a lot of tinkering to get it flying, lemme tell ya! What did the trick was to switch from batch gradient descent to stochastic gradient descent (and removing som bugs/errors). This is really cool!

Considering to rename this project SonOfAnton (for those who get the reference).

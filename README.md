# neural-network
I've built a neural network-[class](https://github.com/magnushelliesen/neural-network/blob/main/neural_network/neural_network.py) (from first principles, only using NumPy) with a train/backpropagation method. I've applied it in a [notebook](https://github.com/magnushelliesen/neural-network/blob/main/neural-network.ipynb), and it works really well (it was a lot of tinkering, lemme tell ya)! What did the trick was to switch from batch gradient descent to stochastic gradient descent. This is really cool!

It also seems to be able to train on the MNIST dataset, it recognizes most but not all digits. There is probably something left to work on with the backpropagation method.

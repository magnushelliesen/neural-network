# neural-network
I've built a neural network-[class](https://github.com/magnushelliesen/neural-network/blob/main/neural_network/neural_network.py) (from first principles, only using NumPy) with a train/backpropagation method. I've applied it in a [notebook](https://github.com/magnushelliesen/neural-network/blob/main/neural-network.ipynb) and it seems to work really well (it was a lot of tinkering, lemme tell ya)! What did the trick was to switch from batch gradient descent to stochastic gradient descent. (It's not really stochastic, but it has the same effect as i show it vectors one by one *as if* it were stochastic.) This is really cool!

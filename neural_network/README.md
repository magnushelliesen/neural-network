# About the backpropagation method (WIP)
The error function used in the output layer is the _Categorical Cross Entropy Loss_,

$$
  L(y,\hat{y})=-\sum_i y_i \log(\hat{y}_i).
$$

The activation function for the output layer is the _Softmax_ function

$$
  \sigma(z_i) = \frac{e^{z_i}}{\sum_{k=1}^K e^{z_k}}.
$$

Together, they give the gradient w.r.t. $z_i$

$$
  \frac{\partial L}{\partial z_i} = \sigma(z_i)-y_i.
$$

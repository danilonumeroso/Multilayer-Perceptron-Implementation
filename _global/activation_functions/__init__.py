import numpy as np

tanh   = lambda x: np.tanh(x)
tanh_g = lambda x: 1 - x**2

sigmoid   = lambda x: 1 / (1 + np.exp(-x))
sigmoid_g = lambda x: x * (1 - x)

softplus   = lambda x: np.log(1+np.exp(x))
softplus_g = lambda x: 1/(1+np.exp(-_softplus_inverse(x)))

_softplus_inverse = lambda x: np.log(np.exp(x)-1)

relu = lambda x: np.maximum(0,x)
def relu_g(x):
    y = x.copy()
    y[y > 0] = 1
    return y

linear   = lambda x: x
linear_g = lambda x: 1

aee = lambda y_pred, y_real: (np.linalg.norm(y_real - y_pred) ** 2).sum() / (2*y_real.shape[0])
mse = lambda y_pred, y_real: (np.square(y_pred - y_real)).mean() / 2

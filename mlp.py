import numpy as np
from ._global.activation_functions import *
from ._global.util import shuffle, batchify
from sklearn.metrics import accuracy_score

class MLP:
    def __init__(
            self,
            n_hidden=10,
            learning_rate=0,
            momentum=0,
            regularization=0,
            max_epoch=200,
            tolerance=0,
            patience=0,
            activation='sigmoid',
            batch_len=200,
            is_classifier=False,
            n_iter_no_early_stop=200
    ):
        self._n_hidden  = n_hidden
        self._eta       = learning_rate
        self._momentum  = momentum
        self._lambda    = regularization
        self._max_epoch = max_epoch
        self._batch_len = batch_len
        self._tolerance = tolerance
        self._patience  = patience

        # Number of layers without considering
        # the input layer.
        self._n_layers  = 2

        self._activation     = activation
        self._is_classifier  = is_classifier

        if self._is_classifier:
            self._out_activation = 'tanh'   # classifier
            self._score_function = 'accuracy'
        else:
            self._out_activation = 'linear' # regressor
            self._score_function = 'aee'

        self._n_iter_no_early_stop = n_iter_no_early_stop

    def __initialize(self, layer_units, n_samples):
        # Weights and Bias initialization
        self.weights_ = []
        self.bias_    = []

        for n_in, n_out in zip(layer_units[:-1], layer_units[1:]):
            weight_init, bias_init = self.__init_weights(n_in, n_out)

            self.weights_.append(weight_init)
            self.bias_.append(bias_init)

        # Previous updates for weights and bias.
        # It'll be used to perform momentum.
        self.updates = [ np.zeros_like(w) for w in self.weights_ ]
        self.updates_bias = [ np.zeros_like(b) for b in self.bias_ ]

        # Activation functions
        if self._activation == 'sigmoid':
            self._transfer = sigmoid
            self._gradient = sigmoid_g
        elif self._activation == 'tanh':
            self._transfer = tanh
            self._gradient = tanh_g
        elif self._activation == 'softplus':
            self._transfer = softplus
            self._gradient = softplus_g
        elif self._activation == 'relu':
            self._transfer = relu
            self._gradient = relu_g
        else:
            raise Exception("Unrecognized activation function provided.")
        # Output function
        if self._out_activation == 'sigmoid':
            self._out = sigmoid
            self._out_gradient = sigmoid_g
        elif self._out_activation == 'tanh':
            self._out = tanh
            self._out_gradient = tanh_g
        elif self._out_activation == 'linear':
            self._out = linear
            self._out_gradient = linear_g
        else:
            raise Exception("Unrecognized out function.")

        if self._batch_len == 'max':
            self._batch_len = n_samples
        else:
            self._batch_len  = np.clip(self._batch_len, 1, n_samples)
        self.loss_curve = []
        self._patience_init = self._patience


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        return {
            "n_hidden": self._n_hidden,
            "learning_rate": self._eta,
            "momentum": self._momentum,
            "regularization": self._lambda,
            "max_epoch": self._max_epoch,
            "tolerance": self._tolerance,
            "patience": self._patience_init,
            "activation": self._activation,
            "batch_len": self._batch_len,
            "n_iter_no_early_stop": self._n_iter_no_early_stop
        }

    def __init_weights(self, n_in, n_out):
        # Gorot's heuristics
        init_bound = np.sqrt(6 / (n_in + n_out))
        
        return np.random.uniform(-init_bound, init_bound, (n_in, n_out)), np.random.uniform(-init_bound, init_bound, n_out)

    def _loss(self, y_pred, y_real):
        # Calculates the loss function
        return aee(y_pred, y_real)

    def _is_best_score(self, score, best_score):
        return \
            (score > best_score and self._score_function == 'accuracy') \
        or \
            (score < best_score and self._score_function in ['aee', 'mse'])
    
    def __forward_pass(self, X, hidden_layer, output_layer):
        hidden_layer = self._transfer(
            np.dot(X, self.weights_[0]) + self.bias_[0]
        )

        output_layer = self._out(
            np.dot(hidden_layer, self.weights_[1]) + self.bias_[1]
        )

        return hidden_layer, output_layer

    def __backpropagate_layer(self, deltas, error, i, layer, weights_gradients, bias_gradients, n_samples):
        deltas[i] = error

        # Computing gradients
        bias_gradients[i] = np.sum(deltas[i], axis=0) / n_samples # Bias case (outputs always 1)
        weights_gradients[i] = np.dot(layer.T, deltas[i])
        weights_gradients[i] += self._lambda * self.weights_[i] # Regularization
        weights_gradients[i] /= n_samples

        return deltas, weights_gradients, bias_gradients

    def __backprop(self, X, y, hidden_layer, output_layer, deltas, weights_gradients, bias_gradients):
        # Performs Backpropagation.
        # Assumes that the forward pass is already done.
        n_samples = X.shape[0]

        hidden = self._n_layers - 1
        
        # From output layer to hidden layer
        deltas, weights_gradients, bias_gradients = \
            self.__backpropagate_layer(
                deltas,
                (output_layer - y) * self._out_gradient(output_layer),
                hidden,
                hidden_layer,
                weights_gradients,
                bias_gradients,
                n_samples
            )

        # From hidden layer to input layer
        deltas, weights_gradients, bias_gradients = \
            self.__backpropagate_layer(
                deltas,
                np.dot(deltas[hidden], self.weights_[hidden].T)
                    * self._gradient(hidden_layer),
                hidden-1,
                X,
                weights_gradients,
                bias_gradients,
                n_samples
            )
        return weights_gradients, bias_gradients

    def __update_weights(self, weights_gradients, bias_gradients):
        weigths_update = [
            - self._eta * dw + self._momentum * update
            for update, dw in zip(self.updates, weights_gradients)
        ]

        bias_update = [
            - self._eta * db + self._momentum * update
            for update, db in zip(self.updates_bias, bias_gradients)
        ]

        # Saving updates for momentum
        self.updates = weigths_update
        self.updates_bias = bias_update

        self.weights_ = [ w + dw for w, dw in zip(self.weights_, self.updates) ]
        self.bias_ = [ b + db for b, db in zip(self.bias_, self.updates_bias) ]

    # Perform training keeping track of the error
    # over an external validation set.
    def fit(self, X, y, X_val=None, y_val = None):

        # Making sure that both validations sets
        # have been passed through the function.
        # Otherwise we'll set them as None.
        if X_val is None or y_val is None:
            X_val = None
            y_val = None

        n_training_samples, n_inputs = X.shape
        self.n_out = y.shape[1]

        layer_units = [n_inputs] + [self._n_hidden] + [self.n_out]

        self.__initialize(layer_units, n_training_samples)

        hidden_layer = np.empty((self._batch_len, self._n_hidden))
        output_layer = np.empty((self._batch_len, self.n_out))

        weights_gradients = [ np.empty((n_in, n_out)) for n_in, n_out in zip(layer_units[:-1], layer_units[1:]) ]
        bias_gradients = [ np.empty(n_out) for n_out in layer_units[1:] ]

        deltas = [ np.empty_like(layer) for layer in [X, hidden_layer] ]

        self.scores = []
        self.best_score = (-np.inf) if self._is_classifier else np.inf
        self.best_loss = np.inf
        self.epoch = 0

        for self.epoch in range(self._max_epoch):
            X, y = shuffle(X, y) # Shuffling the dataset, useful for SGD
            epoch_loss = 0

            for batch_slice in batchify(n_training_samples, self._batch_len):

                batch_size = X[batch_slice].shape[0]
                
                # Forward pass
                hidden_layer, output_layer = self.__forward_pass(X[batch_slice], hidden_layer, output_layer)
                
                # Backpropagation
                weights_gradients, bias_gradients = self.__backprop(
                    X[batch_slice], y[batch_slice], hidden_layer, output_layer,
                    deltas, weights_gradients, bias_gradients
                )

                self.__update_weights(weights_gradients, bias_gradients)
                
                batch_loss = self._loss(output_layer, y[batch_slice])
                values = np.sum(np.array([np.dot(w.ravel(), w.ravel()) for w in self.weights_]))
                batch_loss += (0.5 * self._lambda) * values / batch_size
                epoch_loss += batch_loss * batch_size

            epoch_loss = epoch_loss / n_training_samples
            
            self.loss_curve.append(epoch_loss)

            if self.early_stop(epoch_loss, self.epoch, X_val, y_val, self.scores):
                break

    def early_stop(self, loss, epoch, X_val, y_val, accuracy):

        if X_val is not None and y_val is not None:
            score  = self.score(X_val, y_val)
            
            if self.best_score - score < self._tolerance:
                self._patience = self._patience - 1
            else:
                self._patience = self._patience_init

            if self._is_best_score(score, self.best_score):
                self.best_score = score
                self._best_weights = [ w.copy() for w in self.weights_ ]
                self._best_bias = [ b.copy() for b in self.bias_ ]

            if self._patience == 0:
                self.weights_ = self._best_weights
                self.bias_    = self._best_bias 

            self.scores.append(score)
        else:
            if self.best_loss - loss < self._tolerance:
                self._patience = self._patience -1
            else:
                self._patience = self._patience_init
            
            if loss < self.best_loss:
                self.best_loss = loss
                self._best_weights = [ w.copy() for w in self.weights_ ]
                self._best_bias = [ b.copy() for b in self.bias_ ]

        if epoch <= self._n_iter_no_early_stop:
            # Invalidate stop criterion if it is too early.
            self._patience = self._patience_init
            
        return self._patience == 0

    def score(self, X, y):
        y_pred = self.predict(X)
        if self._is_classifier:
            return accuracy_score(y, y_pred)
        else:
            return aee(y, y_pred)

    def predict(self, X):
        n_samples, n_inputs = X.shape
        layer_units = [n_inputs] + [self._n_hidden] + [self.n_out]

        hidden_layer = [np.empty((n_samples, self._n_hidden))]
        output_layer = [np.empty((n_samples, self.n_out))]

        _, out = self.__forward_pass(X, hidden_layer, output_layer)
        if self._is_classifier:
            return np.around(out)
        else:
            return out

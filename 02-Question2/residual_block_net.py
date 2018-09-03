import numpy as np
import matplotlib.pyplot as plt
from layers import *


class ResidualBlockNet(object):
    """
    A three-layer fully-connected neural network with a residual block. 
    The net has an input dimension of N, a hidden layer dimension of H, 
    a hidden2 layer dimension of H2, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first and second
    fully connected layer.
    The outputs of the second fully-connected layer are the scores for each class.
    """
    def __init__(self, input_size, hidden_size, hidden2_size, output_size, std=1e-4, use_Res=True):
        """
        # Initialize the model. Weights are initialized to small random values and biases are initialized to zero, shape as follows
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, H2)
        b2: Second layer biases; has shape (H2,)
        W3: Third layer weights; has shape (H2, C)
        b3: Third layer biases; has shape(C,)
        Wr: Residual layer weights; has shape (D, H2)
        bb: Residual layer biases; has shape(H2,)
        :param input_size: The dimension D of the input data.
        :param hidden_size: The number of neurons H in the first hidden layer
        :param hidden2_size: The number of neurons H2 in the second hidden layer
        :param output_size: The number of classes C
        :param std: a weighted value for mutiplying weights
        :param use_Res: use residual block shortcut or not
        """

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['W3'] = std * np.random.randn(hidden2_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        self.use_Res = use_Res
        if use_Res == True:
            self.params['Wr'] = std * np.random.randn(input_size, hidden2_size)
            self.params['br'] = np.zeros(hidden2_size)

    def loss(self, X, y=None, reg=0.0):
        """
        # Compute the loss and gradients for a two layer fully connected neural network.
        :param X: Input data of shape (N, D). Each X[i] is a training sample
        :param y: Vector of training labels. y[i] is the label for X[i]
        :param reg: Regularization strength
        :return: loss: Loss (data loss and regularization loss) for this batch of training samples.
                 grads: Dictionary mapping parameter names to gradients of those parameters
                 with respect to the loss function; has the same keys as self.params
        """

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if self.use_Res == True:
            Wr, br = self.params['Wr'], self.params['br']
        N, D = X.shape

        layer1_relu_out, cache1_relu = layer_relu_forward(X, W1, b1)
        layer2_out, cache2 = forward(layer1_relu_out, W2, b2)

        if self.use_Res == True:
            layerr_out, cacher = forward(X, Wr, br)
            layer2_out += layerr_out
        
        layer2_relu_out, cache2_relu = relu_forward(layer2_out)
        layer3_out, cache3 = forward(layer2_relu_out, W3, b3)

        scores = layer3_out

        if y is None:
            return scores

        data_loss, dout = softmax_loss(scores, y)
        reg_loss = 0.5 * reg * (np.sum(W3*W3) + np.sum(W2*W2) + np.sum(W1*W1))

        if self.use_Res == True:
           reg_loss += 0.5 * reg * (np.sum(Wr*Wr))

        loss = data_loss + reg_loss

        grads = {}

        dlayer2_relu_out, dW3, db3 = backward(dout, cache3)
        dlayer2_out = relu_backward(dlayer2_relu_out, cache2_relu)
        dlayer1_relu_out, dW2, db2 = backward(dlayer2_out, cache2)
        dx, dW1, db1 = layer_relu_backward(dlayer1_relu_out, cache1_relu)

        grads['b1'] = db1
        grads['W1'] = dW1 + 1 * reg * W1
        grads['b2'] = db2
        grads['W2'] = dW2 + 1 * reg * W2
        grads['b3'] = db3
        grads['W3'] = dW3 + 1 * reg * W3

        if self.use_Res == True:
            dx, dWr, dbr = backward(dlayer2_out, cacher)
            grads['Wr'] = dWr
            grads['br'] = dbr

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.999,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        # Train this neural network using stochastic gradient descent.
        :param X: A numpy array of shape (N, D) giving training data.
        :param y: A numpy array f shape (N,) giving training labels; y[i] = c
        :param X_val: A numpy array of shape (N_val, D) giving validation data
        :param y_val: A numpy array of shape (N_val,) giving validation labels
        :param learning_rate: learning rate for optimization
        :param learning_rate_decay: factor used to decay the learning rate after each epoch
        :param reg: regularization strength
        :param num_iters: Number of steps to take when optimizing
        :param batch_size: Number of training examples to use per step
        :param verbose: boolean; if true print progress during optimization
        :return: a dict includes stats of loss and accuracy
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] += -grads['W1'] * learning_rate
            self.params['W2'] += -grads['W2'] * learning_rate
            self.params['W3'] += -grads['W3'] * learning_rate
            self.params['b1'] += -grads['b1'] * learning_rate
            self.params['b2'] += -grads['b2'] * learning_rate
            self.params['b3'] += -grads['b3'] * learning_rate
            if self.use_Res == True:
                self.params['Wr'] += -grads['Wr'] * learning_rate
                self.params['br'] += -grads['br'] * learning_rate

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f val_acc %f train_acc %f' % (it, 
                    num_iters, loss, (self.predict(X_val) == y_val).mean(), 
                    (self.predict(X) == y).mean()))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        # Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        :param X: A numpy array of shape (N, D) giving N D-dimensional data points to classify.
        :return: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if self.use_Res == True:
            Wr, br = self.params['Wr'], self.params['br']
        layer1_relu_out, _ = layer_relu_forward(X, W1, b1)
        layer2_out, _ = forward(layer1_relu_out, W2, b2)

        if self.use_Res == True:
            layerr_out, _ = forward(X, Wr, br)
            layer2_out += layerr_out
        
        layer2_relu_out, _ = relu_forward(layer2_out)
        layer3_out, _ = forward(layer2_relu_out, W3, b3)

        scores = layer3_out

        y_pred = np.argmax(scores, axis=1)

        return y_pred

def plot(stats1,stats2):
    """
    # plot loss and accuracy
    :param stats: input from self.train
    :return: plots
    """
    plt.figure(0)
    plt.plot(stats1['loss_history'],'r',label="with residual")
    plt.plot(stats2['loss_history'],'b',label="without residual")
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./plots/history.png")
    plt.figure(1)
    plt.plot(stats1['train_acc_history'], 'r', label="with residual train acc")
    plt.plot(stats2['train_acc_history'], 'b', label="without residual train acc")
    plt.title('Classification train accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.savefig("./plots/train_accuracy.png")
    plt.figure(2)
    plt.plot(stats1['val_acc_history'], 'g', label="with residual train acc")
    plt.plot(stats2['val_acc_history'], 'm', label="without residual train acc")
    plt.title('Classification validation accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.savefig("./plots/val_accuracy.png")
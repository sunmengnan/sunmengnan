import numpy as np


def forward(x, w, b):
    """
    # forward pass for a fully-connected layer
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (d_1 * ... * d_k, M)
    :param b: A numpy array of biases, of shape (M,)
    :return:  out: output, of shape (N, M)
              cache: (x, w, b)
    """
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_ND = np.reshape(x, (N, D))
    out = np.dot(x_ND, w) + b
    cache = (x, w, b)
    return out, cache


def backward(dout, cache):
    """
    # Computes the backward pass for a fully-connected layer
    :param dout: Tuple of:
        x: Input data, of shape (N, d_1, ... d_k)
        w: Weights, of shape (D, M)
    :param cache: x: Input data, of shape (N, d_1, ... d_k); Weights, of shape (D, M)
    :return:dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
            dw: Gradient with respect to w, of shape (D, M)
            db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_ND = np.reshape(x, (N, D))

    dx_ND = np.dot(dout, w.T)
    dx = dx_ND.reshape(x.shape)
    db = np.sum(dout, axis=0)
    dw = np.dot(x_ND.T, dout)

    return dx, dw, db


def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    # Computes the backward pass for a layer of rectified linear units (ReLUs)
    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout
    :return: dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x > 0) * dout
    return dx


def layer_relu_forward(x, w, b):
    """
    # Convenience layer that perorms an layer transform followed by a ReLU
    :param x: Input to the fully connected layer
    :param w: Weights
    :param b: bias
    :return: a tuple of:
            out: Output from the ReLU
            cache: Object to give to the backward pass
    """
    a, fc_cache = forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def layer_relu_backward(dout, cache):
    """
    # Backward pass for the relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = backward(da, fc_cache)
    return dx, dw, db


def softmax_loss(x, y):
    """
    # Computes the loss and gradient for softmax classification
    :param x: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    :param y: Vector of labels, of shape (N,) where y[i] is the label for x[i] 0 <= y[i] < C
    :return: a tuple of:
            loss: Scalar giving the loss
            dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    - regtype: Regularization type: L1 or L2

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    number_of_classes = W.shape[1]
    number_of_train = X.shape[0]
   
    for i in range(number_of_train):
        # softmax --> e^(s_k) / sigma(e^(s_j))
        # s = f(x_i, W) = W * x_i
        s = np.dot(X[i], W)
        sigma = np.sum(np.exp(s))
        for j in range(number_of_classes):
            #L_i = - log(e_(s_y_i)/sigma(e_s_j))
            softmax_loss = np.exp(s[j])/sigma
            #gradient -> chain rule: dL/dW = dL/df * df/dW
            # ▽_w_L = - s_y_i + log(sigma(e_s_j))
            dW[:, j] += (softmax_loss-(j==y[i]))*X[i]
        #L(W) = sigma (L_i (f(x_i, W), y_i)) 
        loss += -np.log(softmax_loss)
    #1/N 
    loss /= number_of_train
    dW /= number_of_train
    
    if regtype == "L2":
        loss += reg*np.sum(W**2)
        dW += reg*W*2 
    else:
        loss += reg*np.sum(np.abs(W))
        dW += reg*np.sign(W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    number_of_train = X.shape[0]
    s = np.dot(X, W)
    sigma = np.sum(np.exp(s))
    sum_of_s = np.sum(np.exp(s), axis=1, keepdims=True)
    softmax = np.exp(s)/sum_of_s
    loss += np.sum(-np.log(softmax[np.arange(number_of_train), y])) 
    loss /= number_of_train
    true_scores = np.zeros_like(softmax)
    true_scores[range(number_of_train), y] = 1
    dW = np.dot(X.T, softmax - true_scores)
    dW /= number_of_train
   
    if regtype == "L2":
        loss += reg*np.sum(W**2)
        dW += reg*W*2 
    else:
        loss += reg*np.sum(np.abs(W))
        dW += reg*np.sign(W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

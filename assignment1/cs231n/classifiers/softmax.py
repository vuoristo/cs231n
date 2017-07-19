import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
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
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    fs = X[i].dot(W)
    fs -= np.max(fs)
    sum_exp = np.sum(np.exp(fs))
    loss += -fs[y[i]] + np.log(sum_exp) + np.sum(W**2)*reg

    for j in range(W.shape[1]):
      dW[:, j] += (np.exp(fs[j]) / sum_exp) * X[i, :]
      if j == y[i]:
        dW[:, j] -= X[i,:]

  loss /= X.shape[0]
  dW /= X.shape[0]

  dW += 2 * reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  fs = X.dot(W)
  fs -= np.reshape(np.amax(fs, axis=1), (-1,1))
  exp_fs = np.exp(fs)
  sum_exp = np.sum(exp_fs, axis=1)
  loss = np.sum(-fs[np.arange(N), y] + np.log(sum_exp)) 
  loss /= N
  loss += np.sum(W**2)*reg

  y_mat = np.zeros(shape = (C, N))
  y_mat[y, np.arange(N)] = 1

  tmp = (exp_fs / np.reshape(sum_exp, (-1,1)))
  dW = X.T.dot(tmp)
  dW -= np.dot(y_mat, X).T
  dW /= N
  dW += 2 * reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


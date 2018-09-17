import numpy as np


def affine_forward(x, theta, theta0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta0)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  reshape_len = np.prod(x.shape[1:])
  out = np.add(np.dot(theta.T, x.reshape(x.shape[0],reshape_len).T).T, theta0.T)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d,h)
    - theta0: biases, of shape (h,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta0: Gradient with respect to theta0, of shape (1,h)
  """
  x, theta, theta0 = cache
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  # 4-5 lines of code expected
  dtheta0 = np.dot(dout.T,np.ones(x.shape[0])).T
  dx = np.dot(dout,theta.T).reshape(x.shape)
  x = x.reshape(x.shape[0],theta.shape[0])
  dtheta = np.dot(dout.T,x).T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x.clip(0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dout[np.where(x<0)] = 0
  dx = dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def elu_forward(x):
  """
  Computes the forward pass for a layer of exponential-linear units (ELUS).
  These have been show to speed up training

  elu(x) = x if x>= 0
        = (e^x - 1) otherwise
  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: Input x, of same shape as dx
  """
  #Preforms RELU opperation on postive data
  out_pos = x.clip(0)
  #Preforms ELU operation on negatve Data
  out_neg = np.exp(x.clip(None, 0)) - 1
  cache = x
  return out_pos + out_neg, cache

def elu_backward(dout, cache):
  """
  Computes the backward pass for a layer of exponential-linear units (ELUS).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = np.multiply(dout, np.exp(x))
  dx[np.where(x>0)] = dout[np.where(x>0)]

  return dx


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # 2 lines of code expected
    mask = np.random.rand(*x.shape) < (1 - p)
    out = np.multiply(mask, x)


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    # 1 line of code expected
    out = x


    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    # 1 line of code expected
    dx = np.multiply(mask, dout)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (F, C, HH, WW)
  - theta0: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  m,C,H,W = x.shape
  F,C,HH,WW = theta.shape
  Hp = 1 + (H + 2*pad - HH) / stride
  Wp = 1 + (W + 2*pad - WW) / stride
  out = np.zeros([m,F,Hp,Wp])
  for filter_ind in range(theta.shape[0]):
    #Reshaped Theta for dot product
    theta_straight = theta[filter_ind][:][:][:].reshape(np.prod(theta.shape[1:]))
    for example_ind in range(x.shape[0]):
      #X(i-th example) padded
      x_pad = np.pad(x[example_ind][:][:][:],((0,0),(pad,pad),(pad,pad)),'constant')
      for x_loc in range(Wp):
        for y_loc in range(Hp):
          x_sample_straight = x_pad[:, (x_loc*stride):(x_loc*stride)+WW,(y_loc*stride):(y_loc*stride)+HH].reshape(np.prod(theta.shape[1:]))
          out[example_ind,filter_ind,x_loc,y_loc] = np.dot(theta_straight,x_sample_straight) + theta0[filter_ind]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)
  return out, cache

def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. (m,F,Hp,Wp)
  - cache: A tuple of (x, theta, theta0, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x (m,C,H,W)
  - dtheta: Gradient with respect to theta (F,C,HH,WW)
  - dtheta0: Gradient with respect to theta0 (F)
  """
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, theta, theta0, conv_param = cache
  m,C,H,W = x.shape
  F,C,HH,WW = theta.shape
  Hp,Wp = dout.shape[2:]
  dx = np.zeros_like(x)
  dtheta = np.zeros_like(theta)
  stride = conv_param['stride']
  pad = conv_param['pad']
  dx = np.pad(np.zeros_like(x),((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
  x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
  dtheta0 = dout.sum(axis = (0,2,3))
  for filter_ind in range(F):
    for example_ind in range(m):
      for dout_x_loc in range(Wp):
        for dout_y_loc in range(Hp):
          ### Dx Caclulations
          slice_ind = (slice(example_ind,example_ind+1),
            slice(0,C),
            slice(dout_x_loc*stride,dout_x_loc*stride+WW),
            slice(dout_y_loc*stride,dout_y_loc*stride+HH))
          dx[slice_ind] += dout[example_ind,filter_ind,dout_x_loc,dout_y_loc]*theta[filter_ind,:,:,:]
          #Dtheta Calcs:
          dtheta[filter_ind,:,:,:] += (dout[example_ind,filter_ind,dout_x_loc,dout_y_loc] * 
            x_pad[example_ind,:,slice(dout_x_loc*stride,dout_x_loc*stride+WW),
            slice(dout_y_loc*stride,dout_y_loc*stride+HH)])
  dx = dx[:,:,pad:dx.shape[2]-pad,pad:dx.shape[3]-pad] 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  m,C,H,W = x.shape
  pool_h = pool_param['pool_height']
  pool_w = pool_param['pool_width']
  stride = pool_param['stride']
  Hp = 1 + (H - pool_h) / stride
  Wp = 1 + (W - pool_w) / stride
  out = np.zeros([m,C,Hp,Wp])
  for example_ind in range(m):
    for layer_ind in range(C):
      x_single = x[example_ind,layer_ind,:,:]
      for x_loc in range(Wp):
        for y_loc in range(Hp):
          out[example_ind,layer_ind,x_loc,y_loc] = np.max(x_single[(x_loc*stride):(x_loc*stride)+pool_w,(y_loc*stride):(y_loc*stride)+pool_h])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  m,C,H,W = x.shape
  pool_h = pool_param['pool_height']
  pool_w = pool_param['pool_width']
  stride = pool_param['stride']
  Hp = 1 + (H - pool_h) / stride
  Wp = 1 + (W - pool_w) / stride
  dx = np.zeros([m,C,H,W])
  for example_ind in range(m):
    for layer_ind in range(C):
      x_single = x[example_ind,layer_ind,:,:]
      for dx_x_loc in range(Wp):
        for dx_y_loc in range(Hp):
          x_loc = dx_x_loc*stride
          y_loc = dx_y_loc*stride
          index_tuple = (slice(example_ind,example_ind+1),
            slice(layer_ind,layer_ind+1),
            slice(x_loc,x_loc+pool_w),
            slice(y_loc,y_loc+pool_h))
          x_exp = x[index_tuple]
          max_value_location = np.unravel_index(np.argmax(x_exp, axis=None), x_exp.shape)
          x_derv = np.zeros_like(x_exp)
          x_derv[max_value_location] = 1
          dx[index_tuple] += dout[example_ind,layer_ind,dx_x_loc,dx_y_loc]*x_derv
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_scores = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx

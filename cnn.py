import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *
from fancy_conv import *


class ThreeLayerFancyNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys      #
    # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
    # weights and biases of the hidden affine layer, and keys 'theta3' and     #
    # 'theta3_0' for the weights and biases of the output affine layer.        #
    ############################################################################
    C,H,W = input_dim
    self.params['theta_loc'] = np.random.normal(0,weight_scale,
        num_filters*C*filter_size**2).reshape((num_filters,C,filter_size,filter_size))
    #theta_loc of size (F,C,H,W)
    #theta_loc_0 of size (F)
    self.params['theta_loc_0'] = np.zeros(num_filters)

    self.params['theta_per'] = np.random.normal(0,weight_scale,
        num_filters*C*filter_size**2).reshape((num_filters,C,filter_size,filter_size))
    #theta_per of size (F,C,H,W)
    #theta_per_0 of size (F)
    self.params['theta_per_0'] = np.zeros(num_filters)

    pad = (filter_size-1)/2
    Hp = 1 + (H + 2* pad - filter_size)/2
    Wp = 1 + (W + 2* pad - filter_size)/2

    #Large Convolution
    self.params['theta_large'] = np.random.normal(0,weight_scale,
        num_filters*num_filters*filter_size**2).reshape((num_filters,
            num_filters,filter_size,filter_size))
    self.params['theta_large_0'] = np.zeros(num_filters)


    self.params['theta_affine_1'] = np.random.normal(0, weight_scale, num_filters*Hp*Wp*hidden_dim).reshape((num_filters*Hp*Wp,hidden_dim))
    self.params['theta_affine_1_0'] = np.zeros(hidden_dim)

    self.params['theta_affine_2'] = np.random.normal(0,weight_scale,hidden_dim*num_classes).reshape((hidden_dim,num_classes))
    self.params['theta_affine_2_0'] = np.random.normal(0,weight_scale,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    #theta1, theta1_0 are filters for local!
    theta_loc, theta_loc_0 = self.params['theta_loc'], self.params['theta_loc_0']
    #theta2, theta2_0 are filters for peripheral
    theta_per, theta_per_0 = self.params['theta_per'], self.params['theta_per_0']
    #theta3, theta3_0 are for large convolution
    theta_large, theta_large_0 = self.params['theta_large'], self.params['theta_large_0']
    #Theta4,5 are for affine layers!
    theta_affine_1, theta_affine_1_0 = self.params['theta_affine_1'], self.params['theta_affine_1_0']
    theta_affine_2, theta_affine_2_0 = self.params['theta_affine_2'], self.params['theta_affine_2_0']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta_loc.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    large_conv_param = {'stride': filter_size, 'pad': 1}
    weave_param = {'num_zeros': 2, 'filter_size': filter_size}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    X_loc, cache_loc = conv_relu_forward(X, theta_loc, theta_loc_0, conv_param)
    X_per, cache_per = conv_relu_forward(X, theta_per, theta_per_0, conv_param)

    X_loc_large, cache_zero = zero_weave_forward(X_loc, weave_param)
    X_per_weave, cache_weave = array_weave_forwards(X_per, weave_param)

    X_combine, cache_combine = array_sum_fowards(X_loc_large, X_per_weave)
    X_2, cache_large_conv = conv_relu_pool_forward(X_combine, theta_large, 
        theta_large_0, large_conv_param, pool_param)
    X_3, cache_affine_1 = affine_relu_forward(X_2, theta_affine_1, theta_affine_1_0)
    scores, cache_affine_2 = affine_forward(X_3, theta_affine_2, theta_affine_2_0)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code
    #Loss Calcs
    m = float(X.shape[0])
    soft_loss, soft_grad = softmax_loss(scores,y)
    reg_loss = 0.5*self.reg*(np.sum(np.square(theta_loc))
        +np.sum(np.square(theta_per))
        +np.sum(np.square(theta_large))
        +np.sum(np.square(theta_affine_2))
        +np.sum(np.square(theta_affine_1))) 
    loss = soft_loss + reg_loss
    #Grad Calcs:

    #Affine Layer 2
    dx, dtheta_affine_2, dtheta_affine_2_0 = affine_backward(soft_grad, cache_affine_2)
    grads['theta_affine_2'] = dtheta_affine_2 + self.reg*(theta_affine_2)
    grads['theta_affine_2_0'] = dtheta_affine_2_0

    #Affine Layer 1
    dx, dtheta_affine_1, dtheta_affine_1_0 = affine_relu_backward(dx, cache_affine_1)
    grads['theta_affine_1'] = dtheta_affine_1 + self.reg*(theta_affine_1)
    grads['theta_affine_1_0'] = dtheta_affine_1_0

    #Large convu layer
    dx, dtheta_large, dtheta_large_0 = conv_relu_pool_backward(dx, cache_large_conv)
    grads['theta_large'] = dtheta_large + self.reg*(theta_large)
    grads['theta_large_0'] = dtheta_large_0 

    #Array Sum:
    #print('conv relu', dx.shape)
    dx = array_sum_backwards(dx, cache_combine)
    #print('array sum', dx.shape)

    #Loc Zero:
    dx_loc = zero_weave_backwards(dx, cache_zero)
    dx_1, dtheta_loc, dtheta_loc_0 = conv_relu_backward(dx_loc, cache_loc)
    grads['theta_loc'] = dtheta_loc + self.reg*(theta_loc)
    grads['theta_loc_0'] = dtheta_loc_0

    #Peripherial Weave:
    dx_per = array_weave_backwards(dx, cache_weave)
    #sprint('weave backwards shape', dx_per.shape)
    dx_2, dtheta_per, dtheta_per_0 = conv_relu_backward(dx_per, cache_per)
    grads['theta_per'] = dtheta_per + self.reg*(theta_per)
    grads['theta_per_0'] = dtheta_per_0



    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

class ThreeLayerNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys      #
    # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
    # weights and biases of the hidden affine layer, and keys 'theta3' and     #
    # 'theta3_0' for the weights and biases of the output affine layer.        #
    ############################################################################
    C,H,W = input_dim
    self.params['theta1'] = np.random.normal(0,weight_scale,num_filters*C*filter_size**2).reshape((num_filters,C,filter_size,filter_size))
    #Theta1 of size (F,C,H,W)
    self.params['theta1_0'] = np.zeros(num_filters)
    #Theta0 of size (F)
    pad = (filter_size-1)/2
    Hp = 1 + (H + 2* pad - filter_size)/2
    Wp = 1 + (W + 2* pad - filter_size)/2
    #After Convolution with Pad and Stride followd by Max Pool
    self.params['theta2'] = np.random.normal(0,weight_scale,num_filters*Hp*Wp*hidden_dim).reshape((num_filters*Hp*Wp,hidden_dim))
    self.params['theta2_0'] = np.zeros(hidden_dim)
    self.params['theta3'] = np.random.normal(0,weight_scale,hidden_dim*num_classes).reshape((hidden_dim,num_classes))
    self.params['theta3_0'] = np.random.normal(0,weight_scale,num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    X_1, cache_1 = conv_relu_pool_forward(X, theta1, theta1_0, conv_param, pool_param)
    X_2, cache_2 = affine_relu_forward(X_1, theta2, theta2_0)
    scores, cache_3 = affine_forward(X_2, theta3, theta3_0)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code
    #Loss Calcs
    m = float(X.shape[0])
    soft_loss, soft_grad = softmax_loss(scores,y)
    reg_loss = 0.5*self.reg*(np.sum(np.square(theta1))+np.sum(np.square(theta2))+np.sum(np.square(theta3))) 
    loss = soft_loss + reg_loss
    #Grad Calcs:
    dx, dtheta3, dtheta3_0 = affine_backward(soft_grad, cache_3)
    grads['theta3'] = dtheta3 + self.reg*(theta3)
    grads['theta3_0'] = dtheta3_0
    dx, dtheta2, dtheta2_0 = affine_relu_backward(dx, cache_2)
    grads['theta2'] = dtheta2 + self.reg*(theta2)
    grads['theta2_0'] = dtheta2_0
    dx, dtheta1, dtheta1_0 = conv_relu_pool_backward(dx, cache_1)
    grads['theta1'] = dtheta1 + self.reg*(theta1)
    grads['theta1_0'] = dtheta1_0 

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

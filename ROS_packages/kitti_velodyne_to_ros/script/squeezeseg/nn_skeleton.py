# Author: Bichen Wu (bichen@berkeley.edu) 02/20/2017
#-*- coding: utf-8 -*-

"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from squeezeseg.utils import util
import numpy as np
import tensorflow as tf


def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32

  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)

  return var




def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """

  var = _variable_on_device(name, shape, initializer, trainable)

  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc):
    self.mc = mc

    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.ph_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # projected lidar points on a 2D spherical surface
    self.ph_lidar_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 5],
        name='lidar_input'
    )

    # A tensor where an element is 1 if the corresponding cell contains an
    # valid lidar measurement. Or if the data is missing, then mark it as 0.
    self.ph_lidar_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
        name='lidar_mask')

    # A tensor where each element contains the class of each pixel
    self.ph_label = tf.placeholder(
        tf.int32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
        name='label')

    # weighted loss for different classes
    self.ph_loss_weight = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
        name='loss_weight')

    # define a FIFOqueue for pre-fetching data
    self.q = tf.FIFOQueue(
        capacity=mc.QUEUE_CAPACITY,
        dtypes=[tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
        shapes=[[],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 5],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL]]
    )

    #     train.py:129 : enqueue_op에 feed_dict가 들어간다
    self.enqueue_op = self.q.enqueue(
        [self.ph_keep_prob, self.ph_lidar_input, self.ph_lidar_mask,
          self.ph_label, self.ph_loss_weight]
    )

    self.keep_prob, self.lidar_input, self.lidar_mask, self.label, \
        self.loss_weight = self.q.dequeue()

    # model parameters
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size

    # flop counter
    self.flop_counter = [] # array of tuple of layer name, flop number

    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.AZIMUTH_LEVEL*mc.ZENITH_LEVEL*3))


  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError


  def _add_output_graph(self):
    """Define how to intepret output."""
    mc = self.mc

    with tf.variable_scope('interpret_output') as scope:
      self.prob = tf.multiply(
          tf.nn.softmax(self.output_prob, dim=-1), self.lidar_mask,
          name='pred_prob')

      self.pred_cls = tf.argmax(self.prob, axis=3, name='pred_cls')

      # add summaries
      for cls_id, cls in enumerate(mc.CLASSES):
        self._activation_summary(self.prob[:, :, :, cls_id], 'prob_'+cls)



  def _add_loss_graph(self):
    """Define the loss operation."""
    mc = self.mc

    with tf.variable_scope('cls_loss') as scope:
      self.cls_loss = tf.identity(
          tf.reduce_sum(
              tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=tf.reshape(self.label, (-1, )),
                  logits=tf.reshape(self.output_prob, (-1, mc.NUM_CLASS))
              ) \
              * tf.reshape(self.lidar_mask, (-1, )) \
              * tf.reshape(self.loss_weight, (-1, ))
          ) / tf.reduce_sum(self.lidar_mask)*mc.CLS_LOSS_COEF,
          name='cls_loss')

      tf.add_to_collection('losses', self.cls_loss)


    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # add loss summaries
    # _add_loss_summaries(self.loss)
    tf.summary.scalar(self.cls_loss.op.name, self.cls_loss)
    tf.summary.scalar(self.loss.op.name, self.loss)



  def _add_train_graph(self):
    """Define the training operation."""
    mc = self.mc

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    #     가변적으로 learning rate가 변한다
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)


    grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)


    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')



  def _add_viz_graph(self):
    """Define the visualization operation."""
    mc = self.mc

    self.label_to_show = tf.placeholder(
        tf.float32, [None, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 3],
        name='label_to_show'
    )

    self.depth_image_to_show = tf.placeholder(
        tf.float32, [None, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
        name='depth_image_to_show'
    )

    self.pred_image_to_show = tf.placeholder(
        tf.float32, [None, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 3],
        name='pred_image_to_show'
    )

    self.show_label = tf.summary.image('label_to_show',
        self.label_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)

    self.show_depth_img = tf.summary.image('depth_image_to_show',
        self.depth_image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)

    self.show_pred = tf.summary.image('pred_image_to_show',
        self.pred_image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)


  def _add_summary_ops(self):
    """Add extra summary operations."""
    mc = self.mc

    iou_summary_placeholders = []
    iou_summary_ops = []

    for cls in mc.CLASSES:
      ph = tf.placeholder(tf.float32, name=cls+'_iou')
      iou_summary_placeholders.append(ph)
      iou_summary_ops.append(
          tf.summary.scalar('Eval/'+cls+'_iou', ph, collections='eval_summary')
      )

    self.iou_summary_placeholders = iou_summary_placeholders
    self.iou_summary_ops = iou_summary_ops



  def _conv_bn_layer(
      self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
      size, stride, padding='SAME', freeze=False, relu=True,
      conv_with_bias=False, stddev=0.001):
    """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
    as constant. Weights have to be initialized from a pre-trained model or
    restored from a checkpoint.

    Args:
      inputs: input tensor
      conv_param_name: name of the convolution parameters
      bn_param_name: name of the batch normalization parameters
      scale_param_name: name of the scale parameters
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      conv_with_bias: whether or not add bias term to the convolution output.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """
    mc = self.mc

    with tf.variable_scope(conv_param_name) as scope:
      channels = inputs.get_shape()[3]

      if mc.LOAD_PRETRAINED_MODEL:
        cw = self.caffemodel_weight
        kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
        if conv_with_bias:
          bias_val = cw[conv_param_name][1]
        mean_val   = cw[bn_param_name][0]
        var_val    = cw[bn_param_name][1]
        gamma_val  = cw[scale_param_name][0]
        beta_val   = cw[scale_param_name][1]
      else:
        kernel_val = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        if conv_with_bias:
          bias_val = tf.constant_initializer(0.0)
        mean_val   = tf.constant_initializer(0.0)
        var_val    = tf.constant_initializer(1.0)
        gamma_val  = tf.constant_initializer(1.0)
        beta_val   = tf.constant_initializer(0.0)

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))

      self.model_params += [kernel]

      if conv_with_bias:
        biases = _variable_on_device('biases', [filters], bias_val,
                                     trainable=(not freeze))
        self.model_params += [biases]

      gamma = _variable_on_device('gamma', [filters], gamma_val,
                                  trainable=(not freeze))
      beta  = _variable_on_device('beta', [filters], beta_val,
                                  trainable=(not freeze))
      mean  = _variable_on_device('mean', [filters], mean_val, trainable=False)
      var   = _variable_on_device('var', [filters], var_val, trainable=False)

      self.model_params += [gamma, beta, mean, var]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, 1, stride, 1], padding=padding,
          name='convolution')

      if conv_with_bias:
        conv = tf.nn.bias_add(conv, biases, name='bias_add')

      conv = tf.nn.batch_normalization(
          conv, mean=mean, variance=var, offset=beta, scale=gamma,
          variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')

      self.model_size_counter.append(
          (conv_param_name, (1+size*size*int(channels))*filters)
      )

      out_shape = conv.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((conv_param_name, num_flops))

      self.activation_counter.append(
          (conv_param_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      if relu:
        return tf.nn.relu(conv)
      else:
        return conv

  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001, bias_init_val=0.0):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, 1, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )

      out_shape = out.get_shape().as_list()

      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]

      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]

      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out


  def _deconv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, init='trunc_norm', relu=True, stddev=0.001):
    """Deconvolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size. An array of size 2 or 1.
      stride: stride. An array of size 2 or 1.
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      init: how to initialize kernel weights. Now accept 'xavier',
          'trunc_norm', 'bilinear'
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    assert len(size) == 1 or len(size) == 2, \
        'size should be a scalar or an array of size 2.'
    assert len(stride) == 1 or len(stride) == 2, \
        'stride should be a scalar or an array of size 2.'
    assert init == 'xavier' or init == 'bilinear' or init == 'trunc_norm', \
        'initi mode not supported {}'.format(init)

    if len(size) == 1:
      size_h, size_w = size[0], size[0]
    else:
      size_h, size_w = size[0], size[1]

    if len(stride) == 1:
      stride_h, stride_w = stride[0], stride[0]
    else:
      stride_h, stride_w = stride[0], stride[1]

    mc = self.mc
    # TODO(bichen): Currently do not support pretrained parameters for deconv
    # layer.

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      in_height = int(inputs.get_shape()[1])
      in_width = int(inputs.get_shape()[2])
      channels = int(inputs.get_shape()[3])

      if init == 'xavier':
          kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
          bias_init = tf.constant_initializer(0.0)
      elif init == 'bilinear':
        assert size_h == 1, 'Now only support size_h=1'
        assert channels == filters, \
            'In bilinear interporlation mode, input channel size and output' \
            'filter size should be the same'
        assert stride_h == 1, \
            'In bilinear interpolation mode, stride_h should be 1'

        kernel_init = np.zeros(
            (size_h, size_w, channels, channels),
            dtype=np.float32)

        factor_w = (size_w + 1)//2
        assert factor_w == stride_w, \
            'In bilinear interpolation mode, stride_w == factor_w'

        center_w = (factor_w - 1) if (size_w % 2 == 1) else (factor_w - 0.5)
        og_w = np.reshape(np.arange(size_w), (size_h, -1))
        up_kernel = (1 - np.abs(og_w - center_w)/factor_w)
        for c in range(channels):
          kernel_init[:, :, c, c] = up_kernel

        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      # Kernel layout for deconv layer: [H_f, W_f, O_c, I_c] where I_c is the
      # input channel size. It should be the same as the channel size of the
      # input tensor. 
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size_h, size_w, filters, channels],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      biases = _variable_on_device(
          'biases', [filters], bias_init, trainable=(not freeze))
      self.model_params += [kernel, biases]

      # TODO(bichen): fix this
      deconv = tf.nn.conv2d_transpose(
          inputs, kernel, 
          [mc.BATCH_SIZE, stride_h*in_height, stride_w*in_width, filters],
          [1, stride_h, stride_w, 1], padding=padding,
          name='deconv')
      deconv_bias = tf.nn.bias_add(deconv, biases, name='bias_add')

      if relu:
        out = tf.nn.relu(deconv_bias, 'relu')
      else:
        out = deconv_bias

      self.model_size_counter.append(
          (layer_name, (1+size_h*size_w*channels)*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*channels*size_h*size_w)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out


  def _pooling_layer(
      self, layer_name, inputs, size, stride, padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, 1, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out


  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001, bias_init_val=0.0):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    mc = self.mc

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            # check the size before layout transform
            assert kernel_val.shape == (hiddens, dim), \
                'kernel shape error at {}'.format(layer_name)
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val, # O x (C*H*W)
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ), # O x C x H x W
                    (2, 3, 1, 0)
                ), # H x W x C x O
                (dim, -1)
            ) # (H*W*C) x O
            # check the size after layout transform
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          try:
            kernel_val = np.transpose(kernel_val, (1,0))
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))

      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      weights = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_device('biases', [hiddens], bias_init)
      self.model_params += [weights, biases]
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
      if relu:
        outputs = tf.nn.relu(outputs, 'relu')

      # count layer stats
      self.model_size_counter.append((layer_name, (dim+1)*hiddens))

      num_flops = 2 * dim * hiddens + hiddens
      if relu:
        num_flops += 2*hiddens
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append((layer_name, hiddens))

      return outputs


  def _recurrent_crf_layer(
      self, layer_name, inputs, bilateral_filters, sizes=[3, 5],
      num_iterations=1, padding='SAME'):
    """Recurrent conditional random field layer. Iterative meanfield inference is
    implemented as a reccurent neural network.

    Args:
      layer_name: layer name
      inputs: input tensor with shape [batch_size, zenith, azimuth, num_class].
      bilateral_filters: filter weight with shape 
          [batch_size, zenith, azimuth, sizes[0]*size[1]-1].
      sizes: size of the local region to be filtered.
      num_iterations: number of meanfield inferences.
      padding: padding strategy
    Returns:
      outputs: tensor with shape [batch_size, zenith, azimuth, num_class].
    """
    assert num_iterations >= 1, 'number of iterations should >= 1'

    mc = self.mc

    with tf.variable_scope(layer_name) as scope:
      # initialize compatibilty matrices
      compat_kernel_init = tf.constant(
          np.reshape(
              np.ones((mc.NUM_CLASS, mc.NUM_CLASS)) - np.identity(mc.NUM_CLASS),
              [1, 1, mc.NUM_CLASS, mc.NUM_CLASS]
          ),
          dtype=tf.float32
      )

      bi_compat_kernel = _variable_on_device(
          name='bilateral_compatibility_matrix',
          shape=[1, 1, mc.NUM_CLASS, mc.NUM_CLASS],
          initializer=compat_kernel_init*mc.BI_FILTER_COEF,
          trainable=True
      )

      self._activation_summary(bi_compat_kernel, 'bilateral_compat_mat')

      angular_compat_kernel = _variable_on_device(
          name='angular_compatibility_matrix',
          shape=[1, 1, mc.NUM_CLASS, mc.NUM_CLASS],
          initializer=compat_kernel_init*mc.ANG_FILTER_COEF,
          trainable=True
      )

      self._activation_summary(angular_compat_kernel, 'angular_compat_mat')

      self.model_params += [bi_compat_kernel, angular_compat_kernel]

      condensing_kernel = tf.constant(
          util.condensing_matrix(sizes[0], sizes[1], mc.NUM_CLASS),
          dtype=tf.float32,
          name='condensing_kernel'
      )

      angular_filters = tf.constant(
          util.angular_filter_kernel(
              sizes[0], sizes[1], mc.NUM_CLASS, mc.ANG_THETA_A**2),
          dtype=tf.float32,
          name='angular_kernel'
      )

      bi_angular_filters = tf.constant(
          util.angular_filter_kernel(
              sizes[0], sizes[1], mc.NUM_CLASS, mc.BILATERAL_THETA_A**2),
          dtype=tf.float32,
          name='bi_angular_kernel'
      )

      for it in range(num_iterations):
        unary = tf.nn.softmax(
            inputs, dim=-1, name='unary_term_at_iter_{}'.format(it))

        ang_output, bi_output = self._locally_connected_layer(
            'message_passing_iter_{}'.format(it), unary,
            bilateral_filters, angular_filters, bi_angular_filters,
            condensing_kernel, sizes=sizes,
            padding=padding
        )

        # 1x1 convolution as compatibility transform
        ang_output = tf.nn.conv2d(
            ang_output, angular_compat_kernel, strides=[1, 1, 1, 1],
            padding='SAME', name='angular_compatibility_transformation')

        self._activation_summary(
            ang_output, 'ang_transfer_iter_{}'.format(it))

        bi_output = tf.nn.conv2d(
            bi_output, bi_compat_kernel, strides=[1, 1, 1, 1], padding='SAME',
            name='bilateral_compatibility_transformation')

        self._activation_summary(
            bi_output, 'bi_transfer_iter_{}'.format(it))

        pairwise = tf.add(ang_output, bi_output,
                          name='pairwise_term_at_iter_{}'.format(it))

        outputs = tf.add(unary, pairwise,
                         name='energy_at_iter_{}'.format(it))

        inputs = outputs

    return outputs



  def _locally_connected_layer(
      self, layer_name, inputs, bilateral_filters,
      angular_filters, bi_angular_filters, condensing_kernel, sizes=[3, 5],
      padding='SAME'):
    """Locally connected layer with non-trainable filter parameters)

    Args:
      layer_name: layer name
      inputs: input tensor with shape 
          [batch_size, zenith, azimuth, num_class].
      bilateral_filters: bilateral filter weight with shape 
          [batch_size, zenith, azimuth, sizes[0]*size[1]-1].
      angular_filters: angular filter weight with shape 
          [sizes[0], sizes[1], in_channel, in_channel].
      condensing_kernel: tensor with shape 
          [size[0], size[1], num_class, (sizes[0]*size[1]-1)*num_class]
      sizes: size of the local region to be filtered.
      padding: padding strategy
    Returns:
      ang_output: output tensor filtered by anguler filter with shape 
          [batch_size, zenith, azimuth, num_class].
      bi_output: output tensor filtered by bilateral filter with shape 
          [batch_size, zenith, azimuth, num_class].
    """
    assert padding=='SAME', 'only support SAME padding strategy'
    assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
        'Currently only support odd filter size.'

    mc = self.mc
    size_z, size_a = sizes
    pad_z, pad_a = size_z//2, size_a//2
    half_filter_dim = (size_z*size_a)//2
    batch, zenith, azimuth, in_channel = inputs.shape.as_list()

    with tf.variable_scope(layer_name) as scope:
      # message passing
      ang_output = tf.nn.conv2d(
          inputs, angular_filters, [1, 1, 1, 1], padding=padding,
          name='angular_filtered_term'
      )

      bi_ang_output = tf.nn.conv2d(
          inputs, bi_angular_filters, [1, 1, 1, 1], padding=padding,
          name='bi_angular_filtered_term'
      )

      condensed_input = tf.reshape(
          tf.nn.conv2d(
              inputs*self.lidar_mask, condensing_kernel, [1, 1, 1, 1], padding=padding,
              name='condensed_prob_map'
          ),
          [batch, zenith, azimuth, size_z*size_a-1, in_channel]
      )

      bi_output = tf.multiply(
          tf.reduce_sum(condensed_input*bilateral_filters, axis=3),
          self.lidar_mask,
          name='bilateral_filtered_term'
      )
      bi_output *= bi_ang_output

    return ang_output, bi_output


  def _bilateral_filter_layer(
      self, layer_name, inputs, thetas=[0.9, 0.01], sizes=[3, 5], stride=1,
      padding='SAME'):
    """Computing pairwise energy with a bilateral filter for CRF.

    Args:
      layer_name: layer name
      inputs: input tensor with shape [batch_size, zenith, azimuth, 2] where the
          last 2 elements are intensity and range of a lidar point.
      thetas: theta parameter for bilateral filter.
      sizes: filter size for zenith and azimuth dimension.
      strides: kernel strides.
      padding: padding.
    Returns:
      out: bilateral filter weight output with size
          [batch_size, zenith, azimuth, sizes[0]*sizes[1]-1, num_class]. Each
          [b, z, a, :, cls] represents filter weights around the center position
          for each class.
    """

    assert padding == 'SAME', 'currently only supports "SAME" padding stategy'
    assert stride == 1, 'currently only supports striding of 1'
    assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
        'Currently only support odd filter size.'

    mc = self.mc
    theta_a, theta_r = thetas
    size_z, size_a = sizes
    pad_z, pad_a = size_z//2, size_a//2
    half_filter_dim = (size_z*size_a)//2
    batch, zenith, azimuth, in_channel = inputs.shape.as_list()

    # assert in_channel == 1, 'Only support input channel == 1'

    with tf.variable_scope(layer_name) as scope:
      condensing_kernel = tf.constant(
          util.condensing_matrix(size_z, size_a, in_channel),
          dtype=tf.float32,
          name='condensing_kernel'
      )

      condensed_input = tf.nn.conv2d(
          inputs, condensing_kernel, [1, 1, stride, 1], padding=padding,
          name='condensed_input'
      )

      # diff_intensity = tf.reshape(
      #     inputs[:, :, :], [batch, zenith, azimuth, 1]) \
      #     - condensed_input[:, :, :, ::in_channel]

      diff_x = tf.reshape(
          inputs[:, :, :, 0], [batch, zenith, azimuth, 1]) \
              - condensed_input[:, :, :, 0::in_channel]
      diff_y = tf.reshape(
          inputs[:, :, :, 1], [batch, zenith, azimuth, 1]) \
              - condensed_input[:, :, :, 1::in_channel]
      diff_z = tf.reshape(
          inputs[:, :, :, 2], [batch, zenith, azimuth, 1]) \
              - condensed_input[:, :, :, 2::in_channel]

      bi_filters = []

      for cls in range(mc.NUM_CLASS):
        theta_a = mc.BILATERAL_THETA_A[cls]
        theta_r = mc.BILATERAL_THETA_R[cls]

        bi_filter = tf.exp(-(diff_x**2+diff_y**2+diff_z**2)/2/theta_r**2)

        bi_filters.append(bi_filter)

      out = tf.transpose(
          tf.stack(bi_filters),
          [1, 2, 3, 4, 0],
          name='bilateral_filter_weights'
      )

    return out


  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary') as scope:
      tf.summary.histogram(layer_name, x)
      tf.summary.scalar(layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(layer_name+'/min', tf.reduce_min(x))

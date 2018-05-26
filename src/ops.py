"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def gru(previous_hidden_state, x, scope=None):


    with tf.variable_scope(scope or "GRU"):

        input_shape = x.get_shape().as_list()
        hidden_layer_shape = previous_hidden_state.get_shape().as_list()

        Wz = tf.get_variable('Wz',shape=[input_shape[1],hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        bz = tf.get_variable('bz',shape=[hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        Wr = tf.get_variable('Wr',shape=[input_shape[1],hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        br = tf.get_variable('br',shape=[hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        Wx = tf.get_variable('Wx',shape=[input_shape[1],hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        Wh = tf.get_variable('Wh',shape=[hidden_layer_shape[0],hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        Wo = tf.get_variable('Wo',shape=[hidden_layer_shape[0],hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())
        bo = tf.get_variable('bo',shape=[hidden_layer_shape[0]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initilizer())

        z = tf.sigmoid(tf.matmul(x, Wz) + bz)
        r = tf.sigmoid(tf.matmul(x, Wr) + br)

        h_ = tf.tanh(tf.matmul(x, Wx) +
                     tf.matmul(previous_hidden_state, Wh) * r)

        current_hidden_state = tf.multiply(
            (1 - z), h_) + tf.multiply(previous_hidden_state, z)

        output = tf.nn.relu(tf.matmul(current_hidden_state, Wo) + bo)

        return current_hidden_state , output

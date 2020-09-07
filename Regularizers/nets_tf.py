from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
import tensorflow.contrib as contrib
from Regularizers.layers_tf import *


def dncnn(input, num_layer=10, output_channels=1):
    input_shape_of_conv_layer = []
    with tf.variable_scope('block1'):
        print('input.[1]:   ',input.shape[1])
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, num_layer):
        with tf.variable_scope('block%d' % layers):
            input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=True)
            output = tf.nn.relu(output)
    with tf.variable_scope('block%d' % num_layer):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return output, input_shape_of_conv_layer

def dncnn_sha5(input, is_training=True, output_channels=1):
    input_shape_of_conv_layer = []
    with tf.variable_scope('block1'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(input, 64, 3, padding='same', dilation_rate=[1,1], activation=tf.nn.relu, use_bias=True)

    with tf.variable_scope('block2'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[2,2], name='conv%d' % 2, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block3'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[3,3], name='conv%d' % 3, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block4'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[2,2], name='conv%d' % 4, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block5'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, output_channels, 3, padding='same', dilation_rate=[1,1])
    return output, input_shape_of_conv_layer


def dncnn_sha7(input, is_training=True, output_channels=1):
    input_shape_of_conv_layer = []
    with tf.variable_scope('block1'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(input, 64, 3, padding='same', dilation_rate=[1,1], activation=tf.nn.relu, use_bias=True)

    with tf.variable_scope('block2'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[2,2], name='conv%d' % 2, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block3'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[3,3], name='conv%d' % 3, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block4'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[4,4], name='conv%d' % 4, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block5'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[3,3], name='conv%d' % 5, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block6'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, 64, 3, padding='same', dilation_rate=[2,2], name='conv%d' % 5, use_bias=True)
        output = tf.nn.relu(output)

    with tf.variable_scope('block7'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, output_channels, 3, padding='same', dilation_rate=[1,1])
    return output, input_shape_of_conv_layer

def dncnn_ori(input, is_training=False, output_channels=1):
    input_shape_of_conv_layer = []
    with tf.variable_scope('block1'):
        print('input.[1]: ' ,input.shape[1])
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return output,input_shape_of_conv_layer

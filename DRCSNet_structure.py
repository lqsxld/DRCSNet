# -*- coding: utf-8 -*-

"""
Created on Nov 13, 2023

@author: Qian Liu
"""

from keras import backend, layers, models
from keras.regularizers import l2

###############################################################################

def _momentum(batch_size):
    if batch_size < 100:
        return 1.0 - 1.0 / batch_size
    else:
        return 0.99


def _bn_relu(x, bn_axis=3, momentum=0.99, name=None):
    x = layers.BatchNormalization(axis=bn_axis, momentum=momentum, epsilon=1e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    return x


def _conv(x, filters, kernel_size=3, strides=1, use_bias=True, groups=1, l2_reg=None, name='_conv'):
    if kernel_size == 1:
        x = layers.Conv2D(filters, kernel_size, kernel_initializer='he_normal', kernel_regularizer=l2_reg,
                          strides=strides, use_bias=use_bias, groups=groups, name=name)(x)
    elif strides == 1:  # and kernel_size > 1
        x = layers.Conv2D(filters, kernel_size, kernel_initializer='he_normal', kernel_regularizer=l2_reg,
                          padding='same', strides=strides, use_bias=use_bias, groups=groups, name=name)(x)
    else:  # kernel_size > 1 and strides > 1:
        padding_size = kernel_size // 2
        x = layers.ZeroPadding2D(padding=((padding_size, padding_size), (padding_size, padding_size)), name=name+'_padding')(x)
        x = layers.Conv2D(filters, kernel_size, kernel_initializer='he_normal', kernel_regularizer=l2_reg,
                          strides=strides, use_bias=use_bias, groups=groups, name=name)(x)

    return x


def _maxpooling(x, area_size=3, strides=2, name='_maxpooling'):
    if area_size == 1:
        x = layers.MaxPooling2D(area_size, strides=strides, name=name)(x)
    elif strides == 1:  # and area_size > 1
        x = layers.MaxPooling2D(area_size, strides=strides, padding='same', name=name)(x)
    else:  # area_size > 1 and strides > 1:
        padding_size = area_size // 2
        x = layers.ZeroPadding2D(padding=((padding_size, padding_size), (padding_size, padding_size)), name=name+'_padding')(x)
        x = layers.MaxPooling2D(area_size, strides=strides, name=name)(x)

    return x


def _block_conv1(x, filters, kernel_size=3, strides=2, use_bias=False, bn_axis=3, l2_reg=None, momentum=0.99, name='conv1'):
    x = _conv(x, filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, l2_reg=l2_reg, name=name+'_conv')
    return x

###############################################################################

def _transition(x, theta=1.0, strides=2, bn_axis=3, use_bias=False, l2_reg=None, momentum=0.99, name=None):
    x = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name=name)

    if theta < 8.001:
        x = _conv(x, round(backend.int_shape(x)[bn_axis] * theta), kernel_size=1, use_bias=use_bias, l2_reg=l2_reg, name=name+'_conv')
    else:
        x = _conv(x, theta, kernel_size=1, use_bias=use_bias, l2_reg=l2_reg, name=name+'_conv')

    if strides > 1:
        x = layers.AveragePooling2D(strides, strides=strides, name=name + '_avgpool')(x)

    return x

###############################################################################

def _residual_33(x, filters, strides=1, bn_axis=3, l2_reg=None, momentum=0.99, name=None):
    x = _conv(x, filters, kernel_size=3, strides=strides, use_bias=False, l2_reg=l2_reg, name=name+'_1_conv')
    x1 = [x]
    x = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name=name + '_1')
    x = _conv(x, filters, kernel_size=3, strides=1, use_bias=False, l2_reg=l2_reg, name=name+'_2_conv')
    x1.append(x)
    return x1


def _residual_133(x, filters, strides=1, bn_axis=3, l2_reg=None, momentum=0.99, name=None):
    x = _conv(x, filters, kernel_size=1, strides=1, use_bias=False, l2_reg=l2_reg, name=name+'_1_conv')
    x = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name=name + '_1')
    x = _conv(x, filters, kernel_size=3, strides=strides, use_bias=False, l2_reg=l2_reg, name=name+'_2_conv')
    x1 = [x]
    x = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name=name + '_2')
    x = _conv(x, filters, kernel_size=3, strides=1, use_bias=False, l2_reg=l2_reg, name=name+'_3_conv')
    x1.append(x)
    return x1

###############################################################################

def _shortcut(filters, res, y, theta=1.0, strides=1, bn_axis=3, l2_reg=None, momentum=0.99, name=None):
    if strides > 1:
        y = layers.Concatenate(axis=bn_axis, name=name+'_2_transition_concat')(y)
        y = [_transition(y, theta=theta, strides=strides, bn_axis=bn_axis, use_bias=False, l2_reg=l2_reg, momentum=momentum, name=name + '_2_transition')]

    if len(y) > 1:
        y_shortcut = layers.Concatenate(axis=bn_axis, name=name+'_1_transition_concat')(y)
    else:
        y_shortcut = y[0]

    shortcut = _transition(y_shortcut, theta=filters, strides=1, bn_axis=bn_axis, use_bias=False, l2_reg=l2_reg, momentum=momentum, name=name + '_1_transition')
    x = layers.Concatenate(axis=bn_axis, name=name+'_1_concat')([shortcut] + res)
    y.extend(res)
    return x, y

###############################################################################

def _block_conv2345(x, filters, sc_filter_num=1, y=None, strides=1, bn_axis=3, res_fun=_residual_33, sc_fun=_shortcut,
                    is_block1_of_conv2=False, use_maxpool=False, conv2_theta=1, theta=1.0, l2_reg=None, momentum=0.99, name=None):
    if is_block1_of_conv2:
        x = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name='conv1')

        if use_maxpool:
            x = _maxpooling(x, area_size=3, strides=strides, name='conv2_maxpooling')
            strides = 1

        x_br = x

        if conv2_theta > 1:
            y = _conv(x, round(backend.int_shape(x)[bn_axis] * conv2_theta), kernel_size=1, use_bias=False, l2_reg=l2_reg, name='conv2_transition_conv')
            y = _bn_relu(y, bn_axis=bn_axis, momentum=momentum, name='conv2_transition')

            if strides > 1:
                y = [layers.AveragePooling2D(strides, strides=strides, name='conv2_transition_avgpool')(y)]
            else:
                y = [y]
        else:
            if strides > 1:
                y = [layers.AveragePooling2D(strides, strides=strides, name='conv2_transition_avgpool')(x)]
            else:
                y = [x]
    else:
        x_br = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name=name + '_residual_0')

    res = res_fun(x_br, filters, strides=strides, bn_axis=bn_axis, l2_reg=l2_reg, momentum=momentum, name=name + '_residual')
    x, y = sc_fun(sc_filter_num * filters, res, y, theta=theta, strides=strides, bn_axis=bn_axis, l2_reg=l2_reg, momentum=momentum, name=name + '_shortcut')
    return x, y

###############################################################################

def _block_stack(x, filters, blocks, y=None, strides=2, mode='33', bn_axis=3, is_conv2=False, use_maxpool=False, theta=1.0, l2_reg=None, momentum=0.99, name=None):
    sc_fun = _shortcut
    sc_filter_num = 3
    conv2_theta = 2

    if mode == '33':
        res_fun = _residual_33
    else: # mode == '133':
        res_fun = _residual_133

    x, y = _block_conv2345(x, filters, sc_filter_num=sc_filter_num, y=y, strides=strides, bn_axis=bn_axis, res_fun=res_fun, sc_fun=sc_fun, is_block1_of_conv2=is_conv2,
                          use_maxpool=use_maxpool, conv2_theta=conv2_theta, theta=theta, l2_reg=l2_reg, momentum=momentum, name=name + '_block1')

    for ii in range(2, blocks+1):
        x, y = _block_conv2345(x, filters, sc_filter_num=sc_filter_num, y=y, strides=1, bn_axis=bn_axis, res_fun=res_fun, sc_fun=sc_fun,
                              theta=theta, l2_reg=l2_reg, momentum=momentum, name=name + '_block' + str(ii))

    return x, y

###############################################################################

def DRCSNet(input_shape, class_num, model_mode='DRCSNet-2-0039-012', batch_size=128, l2_reg=4e-4, channels_mean=None, channels_std=None, weights=None):
    """
        input_shape:    A tuple of input size tuple, which must be (height, width, 3) for channels_last or (3, height, width) for channels_first
        class_num:      The class number for a classification task
        model_mode:     A string, for example, in 'DRCSNet-2-39-12',
                            'DRCSNet-2' denotes that DRCSNet-2 or DRCSNet-3 is used
                            '-0039'     denotes the layer number 39
                            '-012'      denotes the growth rate k=12
        batch_size:     The batch size
        l2_reg:         The parameter value of l2 regularizer for kernel regularizer of convolution
        channels_mean:  The mean value vector of color channels
        channels_std:   The standard deviation value vector of color channels
        weights:        The network weights
                            None        denotes the random initialization
                            A string    denotes the file path of the pretraining weights
    """

    def input_meansub_swap(tensor):
        if backend.image_data_format() == 'channels_last':
            return tensor - channels_mean
        else:
            return backend.permute_dimensions(tensor, (0,2,3,1)) - channels_mean

    def input_normalization_swap(tensor):
        if backend.image_data_format() == 'channels_last':
            return (tensor - channels_mean) / channels_std
        else:
            return (backend.permute_dimensions(tensor, (0,2,3,1)) - channels_mean) / channels_std

    bn_axis = 3
    l2_reg = l2(l2_reg)
    momentum = _momentum(batch_size)
    block_stack = _block_stack
    block_conv1 = _block_conv1
    use_bias = False
    filter_num = int(model_mode[15:])

    if model_mode[10:14] == '0039':
        block_mode = '33'
        blocks_num = [6,6,6]
    elif model_mode[10:14] == '0099':
        block_mode = '33'
        blocks_num = [16,16,16]
    elif model_mode[10:14] == '0102':
        block_mode = '133'
        blocks_num = [11,11,11]
    elif model_mode[10:14] == '0192':
        block_mode = '133'
        blocks_num = [21,21,21]
    elif model_mode[10:14] == '0255':
        block_mode = '133'
        blocks_num = [28,28,28]
    elif model_mode[10:14] == '0120':
        block_mode = '133'
        blocks_num = [4,8,16,11]
    elif model_mode[10:14] == '0162':
        block_mode = '133'
        blocks_num = [4,8,24,17]
    elif model_mode[10:14] == '0171':
        block_mode = '133'
        blocks_num = [4,8,22,22]
    elif model_mode[10:14] == '0201':
        block_mode = '133'
        blocks_num = [4,8,32,22]

    input_tensor = layers.Input(shape=input_shape)

    if model_mode[10:14] in {'0039', '0099', '0102', '0192', '0255'}:  # for CIFAR
        if model_mode[10:14] in {'0039', '0099'}:
            theta = 1.0
        else:
            theta = 0.5

        x = layers.Lambda(input_meansub_swap, output_shape=input_shape, name='input_preprocessing')(input_tensor)
        x = block_conv1(x, filter_num, kernel_size=3, strides=1, bn_axis=bn_axis, use_bias=use_bias, l2_reg=l2_reg, momentum=momentum, name='conv1')
        x, y = block_stack(x, filter_num, blocks_num[0], strides=1, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, is_conv2=True, use_maxpool=False, name='conv2')
        x, y = block_stack(x, filter_num, blocks_num[1], y, strides=2, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, name='conv3')
        x, y = block_stack(x, filter_num, blocks_num[2], y, strides=2, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, name='conv4')
    elif model_mode[10:14] in {'0120', '0162', '0171', '0201'}:  # for ImageNet
        theta = 0.5

        x = layers.Lambda(input_meansub_swap, output_shape=input_shape, name='input_preprocessing')(input_tensor)
        x = block_conv1(x, filter_num,  kernel_size=7, strides=2, bn_axis=bn_axis, use_bias=use_bias, l2_reg=l2_reg, momentum=momentum, name='conv1')
        x, y = block_stack(x, filter_num, blocks_num[0], strides=2, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, is_conv2=True, use_maxpool=True, name='conv2')
        x, y = block_stack(x, filter_num, blocks_num[1], y, strides=2, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, name='conv3')
        x, y = block_stack(x, filter_num, blocks_num[2], y, strides=2, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, name='conv4')
        x, y = block_stack(x, filter_num, blocks_num[3], y, strides=2, mode=block_mode, bn_axis=bn_axis, theta=theta, l2_reg=l2_reg, momentum=momentum, name='conv5')

    y = layers.Concatenate(axis=bn_axis, name='post_transition_concat')(y)
    x = _transition(y, theta=theta, strides=1, bn_axis=bn_axis, use_bias=use_bias, l2_reg=l2_reg, momentum=momentum, name='post_transition')
    x = _bn_relu(x, bn_axis=bn_axis, momentum=momentum, name='post')

    x = layers.GlobalAveragePooling2D(name='classifier_gap')(x)
    output_tensor = layers.Dense(class_num, kernel_initializer='he_normal', kernel_regularizer=l2_reg,
                                 activation='softmax', name='classifier_fc_softmax')(x)

    model = models.Model(input_tensor, output_tensor, name=model_mode)

    # Load weights.
    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

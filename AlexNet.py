# !/usr/bin/python
# -*- coding:utf-8 -*-

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
# import cifar10, cifar10_input

# 显示每一层结构的尺寸
def print_activations(t):
    # 格式为 [batch_size, image_height, image_width, channel_num]
    print(t.op.name, ' ', t.get_shape().as_list())

def inference(images):
    parameters = []

    # 第一个卷积层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(images,
                            kernel,
                            [1, 4, 4, 1],
                            padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True,
                             name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv1)

    # LRN会使前馈和反馈速度大幅度下降，可用可不用
    lrn1 = tf.nn.lrn(conv1,
                     4,
                     bias=1.0,
                     alpha=0.001/9,
                     beta=0.75,
                     name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1], #池化尺寸3*3
                           strides=[1, 2, 2, 1], #步长2*2
                           padding='VALID', #取样不超过边框
                           name='pool1')
    print_activations(pool1)

    # 第二个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool1,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2,
                     4,
                     bias=1.0,
                     alpha=0.001 / 9,
                     beta=0.75,
                     name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],  # 池化尺寸3*3
                           strides=[1, 2, 2, 1],  # 步长2*2
                           padding='VALID',  # 取样不超过边框
                           name='pool2')
    print_activations(pool2)

    # 第三个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool2,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # 第四个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv3,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # 第五个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv4,
                            kernel,
                            [1, 1, 1, 1],
                            padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    # 还需要添加三个全联接层，节点数4096，4096，1000

    # 最大池化层
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],  # 池化尺寸3*3
                           strides=[1, 2, 2, 1],  # 步长2*2
                           padding='VALID',  # 取样不超过边框
                           name='pool5')
    print_activations(pool5)

    return pool5, parameters

def time_tensorflow_run(session, target, info_string):
    # 预热轮数 跳过显存加载，cache命中等问题
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i%10:
                # 每10轮显示当前迭代的时间
                print('%s: step %d, duration=%.3f' %
                      (datetime.now(), i-num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared /num_batches - mn * mn
    sd = math.sqrt(vr)
    # 每轮迭代平均耗时mm，方差sd
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        # 随机选择图片数据
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    time_tensorflow_run(sess, pool5, 'Forward')
    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective, parameters)
    time_tensorflow_run(sess, grad, "Forward-backward")

if __name__ == '__main__':

    batch_size = 32
    num_batches = 100

    run_benchmark()


# !/usr/bin/python
# -*- coding:utf-8 -*-

import time as time
import numpy as np
import tensorflow as tf
import cifar10, cifar10_input
import math

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None: # 加入wl正则惩罚
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # 对所有的loss求和
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == '__main__':

    max_steps = 3000
    batch_size = 128
    data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
    # cifar10.maybe_download_and_extract()

    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                                batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                    data_dir=data_dir,
                                                    batch_size=batch_size)
    # 图片尺寸24*24，RGB三通道
    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

    # 创建卷积层1
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64],
                                        stddev=5e-2,
                                        wl=0.0)  # 不加正则
    kernel1 = tf.nn.conv2d(image_holder,
                           weight1,
                           [1, 1, 1, 1], # 步长全为1
                           padding='SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1], # 尺寸3*3
                           strides=[1, 2, 2, 1], # 步长2*2，增加数据的丰富性
                           padding='SAME')
    # LRN局部响应归一化层 提高泛化能力
    norm1 = tf.nn.lrn(pool1,
                      4,
                      bias=1.0,
                      alpha=0.001/9.0,
                      beta=0.75)

    # 创建卷积层2
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64],
                                        stddev=5e-2,
                                        wl=0.0)  # 不加正则
    kernel2 = tf.nn.conv2d(norm1,
                           weight2,
                           [1, 1, 1, 1],  # 步长全为1
                           padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    # 先做LRN后池化，与第一层顺序不同
    norm2 = tf.nn.lrn(conv2,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75)
    pool2 = tf.nn.max_pool(norm2,
                           ksize=[1, 3, 3, 1],  # 尺寸3*3
                           strides=[1, 2, 2, 1],  # 步长2*2，增加数据的丰富性
                           padding='SAME')

    # 全联接层
    reshape = tf.reshape(pool2, [batch_size, -1])
    # 获取扁平化后的长度
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim, 384],
                                        stddev=0.04,
                                        wl=0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

    # 全联接层2 节点减少一半
    weight4 = variable_with_weight_loss(shape=[384, 192],
                                        stddev=0.04,
                                        wl=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

    # 全联接层3
    weight5 = variable_with_weight_loss(shape=[192, 10],
                                        stddev=1/192.0,
                                        wl=0.0) # 不加正则
    bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
    logits = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

    # 计算损失
    loss = loss(logits, label_holder)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    # 输出分数最高的那一类准确率
    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # 使用16个线程加速计算
    tf.train.start_queue_runners()

    # 训练
    for step in range(max_steps):
        start_time = time.time()
        # 获得一个batch的训练数据
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict={image_holder: image_batch, label_holder: label_batch})

        duration = time.time() - start_time
        if step%10 == 0:
            examples_per_sec = batch_size/duration
            sec_per_batch = float(duration)
            format_str=('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str %(step, loss_value, examples_per_sec, sec_per_batch))

    # 测试集
    num_examples = 10000
    num_iter = int(math.ceil(num_examples/batch_size))
    true_count = 0;
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op],
                               feed_dict={image_holder: image_batch, label_holder: label_batch})
        true_count += np.sum(predictions)
        step +=1

    precision = true_count/total_sample_count
    print("精度为", precision)
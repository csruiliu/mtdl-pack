import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
from nets.mobilenet import mobilenet_v2

def filter_net(images1,images2):
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints1 = resnet_v2.resnet_v2_50(images1, 1001, is_training=False, reuse=tf.AUTO_REUSE)
        print("res")
        print(net.name)

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints2 = mobilenet_v2.mobilenet(images2, reuse=tf.AUTO_REUSE)
        print("mobile")
        print(logits.name)

    return endpoints1["predictions"], endpoints2["Predictions"]

def mobile_net(images):
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints = mobilenet_v2.mobilenet(images, reuse=tf.AUTO_REUSE)
        print("mobile")
        print(logits.name)
    return endpoints["Predictions"]

def res_net(images):
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(images, 1001, is_training=False, reuse=tf.AUTO_REUSE)
        print("res")
        print(net.name)
    return endpoints["predictions"]

import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')
import os

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
from nets.mobilenet import mobilenet_v2
import timeit
from mr_model import res_net
from mr_model import mobile_net
from tensorflow.contrib.lookup import HashTable

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--grow", action="store_true", help="allowing GPU memory growth")
parser.add_argument("-l", "--loop", type=int, default=1, help="average time")
args = parser.parse_args()

loop_num = args.loop

tf.reset_default_graph()

img_w = 224
img_h = 224
img_size = img_w * img_h
img_channel = 3

net = tf.Graph()
with net.as_default():
    file_input_r = tf.placeholder(tf.string, ())
    image_r = tf.image.decode_jpeg(tf.read_file(file_input_r))
    images_r = tf.expand_dims(image_r, 0)
    images_r = tf.cast(images_r, tf.float32) / 128.  - 1
    images_r.set_shape((None, None, None, 3))
    images_r = tf.image.resize_images(images_r, (224, 224))
    output_r = res_net(images_r)

    saver_r = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))

    if args.grow:
        print("Growth GPU Memory Mode:")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=net, config=config) as sess:
            checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
            saver_r.restore(sess,  checkpoint_r)
            pred_r = sess.run(output_r, feed_dict={file_input_r: '../data/img/img1.jpg'})
            label_map = imagenet.create_readable_names_for_imagenet_labels()
            print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            time = 0
            for n in range(loop_num):
                for i in range(1,101):
                    img_name = '../data/img/img'+str(i)+'.jpg'
                    start = timeit.default_timer()
                    pred_r = sess.run(output_r, feed_dict={file_input_r: img_name})
                    stop = timeit.default_timer()
                    time += stop - start
                    print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            print(" time ", time/float(loop_num))
    else:
        with tf.Session(graph=net) as sess:
            checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
            saver_r.restore(sess,  checkpoint_r)
            pred_r = sess.run(output_r, feed_dict={file_input_r: '../data/img/img1.jpg'})
            label_map = imagenet.create_readable_names_for_imagenet_labels()
            print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            time = 0
            for n in range(loop_num):
                for i in range(1,101):
                    img_name = '../data/img/img'+str(i)+'.jpg'
                    start = timeit.default_timer()
                    pred_r = sess.run(output_r, feed_dict={file_input_r: img_name})
                    stop = timeit.default_timer()
                    time += stop - start
                    print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            print(" time ", time/float(loop_num))
        
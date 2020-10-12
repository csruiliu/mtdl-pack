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
    file_input_m = tf.placeholder(tf.string, ())
    image_m = tf.image.decode_jpeg(tf.read_file(file_input_m))
    images_m = tf.expand_dims(image_m, 0)
    images_m = tf.cast(images_m, tf.float32) / 128.  - 1
    images_m.set_shape((None, None, None, 3))
    images_m = tf.image.resize_images(images_m, (224, 224))
    output_m = res_net(images_m)

    saver_m = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MobilenetV2'))

    if args.grow:
        print("Growth GPU Memory Mode:")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=net, config=config) as sess:
            checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"
            saver_m.restore(sess,  checkpoint_m)
            pred_m = sess.run(output_m, feed_dict={file_input_m: '../data/img/img1.jpg'})
            label_map = imagenet.create_readable_names_for_imagenet_labels()
            print("Resnet: Top 1 prediction: ", pred_m.argmax(),label_map[pred_m.argmax()], pred_m.max())
            time = 0
            for n in range(loop_num):
                for i in range(1,101):
                    img_name = '../data/img/img'+str(i)+'.jpg'
                    start = timeit.default_timer()
                    pred_m = sess.run(output_m, feed_dict={file_input_m: img_name})
                    stop = timeit.default_timer()
                    time += stop - start
                    print("Resnet: Top 1 prediction: ", pred_m.argmax(),label_map[pred_m.argmax()], pred_m.max())
            print(" time ", time/float(loop_num))
    else:
        with tf.Session(graph=net) as sess:
            checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"
            saver_m.restore(sess,  checkpoint_m)
            pred_m = sess.run(output_m, feed_dict={file_input_m: '../data/img/img1.jpg'})
            label_map = imagenet.create_readable_names_for_imagenet_labels()
            print("Resnet: Top 1 prediction: ", pred_m.argmax(),label_map[pred_m.argmax()], pred_m.max())
            time = 0
            for n in range(loop_num):
                for i in range(1,101):
                    img_name = '../data/img/img'+str(i)+'.jpg'
                    start = timeit.default_timer()
                    pred_m = sess.run(output_m, feed_dict={file_input_m: img_name})
                    stop = timeit.default_timer()
                    time += stop - start
                    print("Resnet: Top 1 prediction: ", pred_m.argmax(),label_map[pred_m.argmax()], pred_m.max())
            print(" time ", time/float(loop_num))
        
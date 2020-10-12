import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
import timeit

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--grow", action="store_true", help="Allowing GPU memory growth")
args = parser.parse_args()

tf.reset_default_graph()

file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
    net, endpoints = resnet_v2.resnet_v2_50(images, 1001, is_training=False)

saver = tf.train.Saver()

checkpoint = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"

if args.grow:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint)
        start = timeit.default_timer()
        x = endpoints['predictions'].eval(feed_dict={file_input: '../data/img.jpg'})
        stop = timeit.default_timer()
        print('Inference Time: ', stop - start)
else: 
    with tf.Session() as sess:
        saver.restore(sess,  checkpoint)
        start = timeit.default_timer()
        x = endpoints['Predictions'].eval(feed_dict={file_input: '../data/img.jpg'})
        stop = timeit.default_timer()
        print('Inference Time: ', stop - start)

label_map = imagenet.create_readable_names_for_imagenet_labels()
print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())


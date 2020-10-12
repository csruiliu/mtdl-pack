import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
from nets.mobilenet import mobilenet_v2
import timeit
from mr_model import res_net
from mr_model import mobile_net

tf.reset_default_graph()

img_w = 224
img_h = 224
img_size = img_w * img_h
img_channel = 3

net = tf.Graph()
with net.as_default():
    file_input_m = tf.placeholder(tf.string, ())
    file_input_r = tf.placeholder(tf.string, ())
    image_m = tf.image.decode_jpeg(tf.read_file(file_input_m))
    image_r = tf.image.decode_jpeg(tf.read_file(file_input_r))
    images_m = tf.expand_dims(image_m, 0)
    images_r = tf.expand_dims(image_r, 0)
    images_m = tf.cast(images_m, tf.float32) / 128.  - 1
    images_r = tf.cast(images_r, tf.float32) / 128.  - 1
    images_m.set_shape((None, None, None, 3))
    images_r.set_shape((None, None, None, 3))
    images_m = tf.image.resize_images(images_m, (224, 224))
    images_r = tf.image.resize_images(images_r, (224, 224))
    output_m = mobile_net(images_m)
    output_r = res_net(images_r)

    saver_r = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
    saver_m = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MobilenetV2') )

    with tf.Session(graph=net) as sess:
        checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
        saver_r.restore(sess,  checkpoint_r)
        checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"
        saver_m.restore(sess,  checkpoint_m)
        for i in range(20):
            if i == 19:
                start = timeit.default_timer()
                x = sess.run([output_m], feed_dict={file_input_m: '../data/airplane224.jpg'})
                y = sess.run([output_r], feed_dict={file_input_r: '../data/panda224.jpg'})
                stop = timeit.default_timer()
                print("time", stop-start)
            else:
                x = sess.run([output_m], feed_dict={file_input_m: '../data/airplane224.jpg'})
                y = sess.run([output_r], feed_dict={file_input_r: '../data/panda224.jpg'})

        label_map = imagenet.create_readable_names_for_imagenet_labels()
        print("Resnet: Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
        print("Mobilenet: Top 1 prediction: ", y.argmax(),label_map[y.argmax()], y.max())
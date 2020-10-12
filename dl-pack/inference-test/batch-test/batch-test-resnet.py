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
parser.add_argument("-d", "--dataset", type=int, default=100, choices=[10, 100, 1000, 10000], help="test on various dataset")
parser.add_argument("-b", "--batch", type=int, default=10, help="batch size")
parser.add_argument("-g", "--grow", action="store_true", help="allowing GPU memory growth")
parser.add_argument("-l", "--loop", type=int, default=1, help="average time")
args = parser.parse_args()

dataset_size = args.dataset
loop_num = args.loop
batch_size = args.batch
batch_num = dataset_size / batch_size

tf.reset_default_graph()

img_w = 224
img_h = 224
img_size = img_w * img_h
img_channel = 3

net = tf.Graph()
with net.as_default():
    img_path = tf.placeholder(tf.string, ())
    batch = []
    pred_list = []
    for i in range(1,batch_size+1):
        img_name = 'img'+str(i)+'.jpg'
        image_r = tf.image.decode_jpeg(tf.read_file(img_path+'/'+img_name))
        images_r = tf.expand_dims(image_r, 0)
        images_r = tf.cast(images_r, tf.float32) / 128.  - 1
        images_r.set_shape((None, None, None, 3))
        images_r = tf.image.resize_images(images_r, (224, 224))
        batch.append(images_r)
    for j in range(batch_size):
        images_r = batch[j]
        output_r = res_net(images_r)
        pred_list.append(output_r)

    saver_r = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))

    if args.grow:
        print("Growth GPU Memory Mode:")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=net, config=config) as sess:
            checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
            saver_r.restore(sess,  checkpoint_r)
            predlist_r = sess.run(pred_list, feed_dict={img_path: '../data/img/img1'})
            #label_map = imagenet.create_readable_names_for_imagenet_labels()
            #print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            time = 0
            for i in range(loop_num):
                for j in range(1, int(batch_num)+1):
                    start = timeit.default_timer()
                    folder_name = "../data/img/img"+str(j)
                    print(folder_name)
                    predlist_r = sess.run(pred_list, feed_dict={img_path: folder_name})
                    stop = timeit.default_timer()
                    time += stop - start
                    #for k in range(batch_size):
                        #pred_r = predlist_r[k] 
                        #print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            print(" time ", time/float(loop_num))
    else:
        print("Non-growth GPU Memory Mode:")
        with tf.Session(graph=net) as sess:
            checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
            saver_r.restore(sess,  checkpoint_r)
            predlist_r = sess.run(pred_list, feed_dict={img_path: '../data/img/img1'})
            #label_map = imagenet.create_readable_names_for_imagenet_labels()
            #print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            time = 0
            for i in range(loop_num):
                for j in range(1, int(batch_num)+1):
                    start = timeit.default_timer()
                    folder_name = "../data/img/img"+str(j)
                    print(folder_name)
                    predlist_r = sess.run(pred_list, feed_dict={img_path: folder_name})
                    stop = timeit.default_timer()
                    time += stop - start
                    #for k in range(batch_size):
                    #    pred_r = predlist_r[k]
                    #    print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            print(" time ", time/float(loop_num))
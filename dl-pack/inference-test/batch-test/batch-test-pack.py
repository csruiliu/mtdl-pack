import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets import resnet_v2
from datasets import imagenet
from nets.mobilenet import mobilenet_v2
import timeit
from mr_model import filter_net

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
    img_path_r = tf.placeholder(tf.string, ())
    img_path_m = tf.placeholder(tf.string, ())

    batch_r = []
    batch_m = []
    pred_list_m = []
    pred_list_r = []

    for i in range(1,batch_size+1):
        img_name = 'img'+str(i)+'.jpg'
        
        image_r = tf.image.decode_jpeg(tf.read_file(img_path_r+'/'+img_name))
        images_r = tf.expand_dims(image_r, 0)
        images_r = tf.cast(images_r, tf.float32) / 128.  - 1
        images_r.set_shape((None, None, None, 3))
        images_r = tf.image.resize_images(images_r, (224, 224))
        batch_r.append(images_r)

        image_m = tf.image.decode_jpeg(tf.read_file(img_path_m+'/'+img_name))
        images_m = tf.expand_dims(image_m, 0)
        images_m = tf.cast(images_m, tf.float32) / 128.  - 1
        images_m.set_shape((None, None, None, 3))
        images_m = tf.image.resize_images(images_m, (224, 224))
        batch_m.append(images_m)

    for j in range(batch_size):
        images_r = batch_r[j]
        images_m = batch_m[j]
        output_r, output_m = filter_net(images_r, images_m)
        pred_list_r.append(output_r)
        pred_list_m.append(output_m)
            
    saver_r = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
    saver_m = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MobilenetV2'))

    if args.grow:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=net, config=config) as sess:
            checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
            saver_r.restore(sess, checkpoint_r)
            checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"
            saver_m.restore(sess, checkpoint_m)
            pred_r, pred_m = sess.run([pred_list_r,pred_list_m], feed_dict={img_path_r: '../data/img/img1', img_path_m: '../data/img/img2'})
            #label_map = imagenet.create_readable_names_for_imagenet_labels()
            #print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            #print("Mobilenet: Top 1 prediction: ", pred_m.argmax(),label_map[pred_m.argmax()], pred_m.max())
            time = 0
            for i in range(loop_num):
                for j in range(1, int(batch_num)+1):
                    #img_name_r = 'img'+str(j)+'.jpg'
                    #img_name_m = 'img'+str(j+1)+'.jpg'
                    start = timeit.default_timer()
                    folder_name_r = "../data/img/img"+str(j)
                    folder_name_m = "../data/img/img"+str(1001-j)
                    print(folder_name_r)
                    x, y = sess.run([pred_list_r,pred_list_m], feed_dict={img_path_r: folder_name_r, img_path_m: folder_name_m})
                    stop = timeit.default_timer()
                    time += stop - start
                    #label_map = imagenet.create_readable_names_for_imagenet_labels()
                    #print("Resnet: Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
                    #print("Mobilenet: Top 1 prediction: ", y.argmax(),label_map[y.argmax()], y.max())
            print("average time: ", time / float(loop_num))
    else:
        with tf.Session(graph=net) as sess:
            checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"
            saver_r.restore(sess, checkpoint_r)
            checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"
            saver_m.restore(sess, checkpoint_m)
            pred_r, pred_m = sess.run([pred_list_r,pred_list_m], feed_dict={img_path_r: '../data/img/img1', img_path_m: '../data/img/img2'})
            #label_map = imagenet.create_readable_names_for_imagenet_labels()
            #print("Resnet: Top 1 prediction: ", pred_r.argmax(),label_map[pred_r.argmax()], pred_r.max())
            #print("Mobilenet: Top 1 prediction: ", pred_m.argmax(),label_map[pred_m.argmax()], pred_m.max())
            time = 0
            for i in range(loop_num):
                for j in range(1, int(batch_num)+1):
                    #img_name_r = 'img'+str(j)+'.jpg'
                    #img_name_m = 'img'+str(j+1)+'.jpg'
                    start = timeit.default_timer()
                    folder_name_r = "../data/img/img"+str(j)
                    folder_name_m = "../data/img/img"+str(1001-j)
                    x, y = sess.run([pred_list_r,pred_list_m], feed_dict={img_path_r: folder_name_r, img_path_m: folder_name_m})
                    stop = timeit.default_timer()
                    time += stop - start
                    #label_map = imagenet.create_readable_names_for_imagenet_labels()
                    #print("Resnet: Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
                    #print("Mobilenet: Top 1 prediction: ", y.argmax(),label_map[y.argmax()], y.max())
            print("average time: ", time / float(loop_num))
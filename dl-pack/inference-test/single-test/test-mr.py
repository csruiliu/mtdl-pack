import sys
sys.path.append('/home/ruiliu/Development/mtml-tf/models/research/slim')
sys.path.append('/home/ruiliu/Development/mtml-tf/models/official')

import tensorflow as tf
from nets.mobilenet import mobilenet_v2
from nets import resnet_v2
from datasets import imagenet
import timeit


tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

mobile_net = tf.Graph()
with mobile_net.as_default():

    file_input = tf.placeholder(tf.string, ())
    image = tf.image.decode_jpeg(tf.read_file(file_input))

    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128.  - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (224, 224))

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits_m, endpoints_m = mobilenet_v2.mobilenet(images)

    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()

    saver_m = tf.train.Saver()

    checkpoint_m = "/home/ruiliu/Development/tf-exp/ckpts/mobilenet2/mobilenet_v2_1.0_224.ckpt"

with tf.Session(graph=mobile_net) as sess_m:
    saver_m.restore(sess_m, checkpoint_m)
    count = 0
    for i in range(20):
        if i == 19:
            start = timeit.default_timer()
            x = endpoints_m['Predictions'].eval(feed_dict={file_input: '../data/airplane224.jpg'})
            stop = timeit.default_timer()
            infertime_m = stop-start
        else:
            x = endpoints_m['Predictions'].eval(feed_dict={file_input: '../data/airplane224.jpg'})
    print('Inference Time: ', infertime_m)

label_map = imagenet.create_readable_names_for_imagenet_labels()
print("Top 1 prediction[mobilenet]: ", x.argmax(),label_map[x.argmax()], x.max())

tf.reset_default_graph()

resnet_net = tf.Graph()
with resnet_net.as_default():
    file_input = tf.placeholder(tf.string, ())
    image = tf.image.decode_jpeg(tf.read_file(file_input))

    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128.  - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (224, 224))

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net_r, endpoints_r = resnet_v2.resnet_v2_50(images, 1001, is_training=False)

    saver_r = tf.train.Saver()

    checkpoint_r = "/home/ruiliu/Development/tf-exp/ckpts/resnet2/resnet_v2_50.ckpt"

with tf.Session(graph=resnet_net) as sess_r:
    saver_r.restore(sess_r, checkpoint_r)
    x = endpoints_r['predictions'].eval(feed_dict={file_input: '../data/airplane224.jpg'})
    for i in range(20):
        if i == 19:
            start = timeit.default_timer()
            x = endpoints_r['predictions'].eval(feed_dict={file_input: '../data/airplane224.jpg'})
            stop = timeit.default_timer()
            infertime_r = stop - start
        else:
            x = endpoints_r['predictions'].eval(feed_dict={file_input: '../data/airplane224.jpg'})
    print('Inference Time: ',  infertime_r)

label_map = imagenet.create_readable_names_for_imagenet_labels()
print("Top 1 prediction[resnet]: ", x.argmax(),label_map[x.argmax()], x.max())

print('Total Inference Time: ', infertime_r + infertime_m)


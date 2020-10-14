import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from timeit import default_timer as timer
import os
import sys
sys.path.append(os.path.abspath(".."))

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from models.model_importer import ModelImporter
from utils.utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot


def build_model():
    model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
    dm = ModelImporter(train_model_type, str(model_name_abbr.pop()), train_layer_num, img_height, img_width, num_channel,
                       num_class, train_batch_size, train_optimizer, train_learn_rate, train_activation, False)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(features, is_training=True)
    train_step = model_entity.train(model_logit, labels)

    return train_step


def train_model():
    step_time = 0
    step_count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    overall_time_start = timer()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // train_batch_size

        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                if i != 0:
                    start_time = timer()

                batch_offset = i * train_batch_size
                batch_end = (i + 1) * train_batch_size
                if train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature[batch_offset:batch_end]

                train_label_batch = train_label[batch_offset:batch_end]

                if use_tf_timeline:
                    profile_path = cfg_path.profile_path
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch},
                             options=run_options, run_metadata=run_metadata)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(profile_path + '/' + str(train_model_type) + '-' + str(train_batch_size) + '-'
                                      + str(i) + '.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                else:
                    sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

                if i != 0:
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

    overall_time_end = timer()

    overall_time = overall_time_end - overall_time_start

    print('overall training time (s):{}, average step time (ms):{}'.format(overall_time, step_time / step_count * 1000))


if __name__ == '__main__':

    ##########################################
    # Hyperparameters read from config
    ##########################################

    rand_seed = cfg_para.single_rand_seed

    num_epoch = cfg_para.single_num_epoch
    train_optimizer = cfg_para.single_opt
    train_layer_num = cfg_para.single_num_layer
    train_learn_rate = cfg_para.single_learning_rate
    train_activation = cfg_para.single_activation
    train_model_type = cfg_para.single_model_type
    train_batch_size = cfg_para.single_batch_size
    train_dataset = cfg_para.single_train_dataset
    use_tf_timeline = cfg_para.single_use_tb_timeline
    use_cpu = cfg_para.single_use_cpu

    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #################################################
    # Hyperparameters due to dataset
    #################################################

    img_width = 0
    img_height = 0
    num_class = 0
    num_channel = 0

    if train_dataset == 'imagenet':
        train_img_path = cfg_path.imagenet_t50k_img_raw_path
        train_label_path = cfg_path.imagenet_t50k_label_path
        test_img_path = cfg_path.imagenet_t1k_img_raw_path
        test_label_path = cfg_path.imagenet_t1k_label_path

        img_width = cfg_para.img_width_imagenet
        img_height = cfg_para.img_height_imagenet
        num_channel = cfg_para.num_channels_rgb
        num_class = cfg_para.num_class_imagenet

        train_label = load_imagenet_labels_onehot(train_label_path, num_class)
        test_label = load_imagenet_labels_onehot(test_label_path, num_class)

    elif train_dataset == 'cifar10':
        img_width = cfg_para.img_width_cifar10
        img_height = cfg_para.img_height_cifar10
        num_channel = cfg_para.num_channels_rgb
        num_class = cfg_para.num_class_cifar10

        train_feature, train_label, test_feature, test_label = load_cifar10_keras()

    elif train_dataset == 'mnist':
        img_width = cfg_para.img_width_mnist
        img_height = cfg_para.img_height_mnist
        num_channel = cfg_para.num_channels_bw
        num_class = cfg_para.num_class_mnist

        train_img_path = cfg_path.mnist_train_img_path
        train_label_path = cfg_path.mnist_train_label_path
        test_img_path = cfg_path.mnist_test_10k_img_path
        test_label_path = cfg_path.mnist_test_10k_label_path

        train_feature = load_mnist_image(train_img_path)
        train_label = load_mnist_image(test_img_path)
        test_feature = load_mnist_image(test_img_path)
        test_label = load_mnist_label_onehot(test_label_path)

    else:
        raise ValueError('Training Dataset is invaild, only support mnist, cifar10, imagenet')

    ###########################################################
    # Build and train model due to input dataset
    ###########################################################

    train_op = build_model()

    train_model()

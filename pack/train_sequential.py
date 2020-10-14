import tensorflow as tf
from tensorflow.python.client import timeline
import os
from multiprocessing import Process
import numpy as np
from timeit import default_timer as timer
import sys
sys.path.append(os.path.abspath(".."))

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from models.model_importer import ModelImporter
from utils.utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot


def build_model():
    train_collection = list()

    model_name_abbr = np.random.choice(rand_seed, len(train_model_type_list), replace=False).tolist()

    for midx, mvalue in enumerate(train_model_type_list):
        dm = ModelImporter(mvalue, str(model_name_abbr.pop()), train_layer_num_list[midx], img_width, img_height,
                           num_channel, num_class, train_batch_size_list[midx], train_optimizer_list[midx],
                           train_learn_rate_list[midx], train_activation_list[midx], batch_padding=False)

        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(names['features' + str(midx)], is_training=True)
        train_op = model_entity.train(model_logit, names['labels' + str(midx)])
        train_collection.append(train_op)

    return train_collection


def train_model(train_step_arg, batch_size_arg, tidx_arg):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // batch_size_arg

        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                batch_offset = i * batch_size_arg
                batch_end = (i + 1) * batch_size_arg
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
                    sess.run(train_step_arg, feed_dict={names['features' + str(tidx_arg)]: train_feature_batch,
                                                        names['labels' + str(tidx_arg)]: train_label_batch},
                             options=run_options, run_metadata=run_metadata)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(profile_path + '/' + str(train_model_type_list[tidx_arg]) + '-'
                                      + str(batch_size_arg) + '-' + str(i) + '.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                else:
                    sess.run(train_step_arg, feed_dict={names['features' + str(tidx_arg)]: train_feature_batch,
                                                        names['labels' + str(tidx_arg)]: train_label_batch})


if __name__ == '__main__':

    ##########################################
    # Hyperparameters read from config
    ##########################################

    rand_seed = cfg_para.single_rand_seed

    train_model_type_list = cfg_para.pack_model_type
    train_optimizer_list = cfg_para.pack_opt
    train_layer_num_list = cfg_para.pack_num_layer
    train_activation_list = cfg_para.pack_activation
    train_batch_size_list = cfg_para.pack_batch_size
    train_learn_rate_list = cfg_para.pack_learning_rate

    num_epoch = cfg_para.pack_num_epoch
    train_dataset = cfg_para.pack_train_dataset
    use_tf_timeline = cfg_para.pack_use_tb_timeline
    use_cpu = cfg_para.pack_use_cpu

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

    #########################
    # Build and Train
    #########################

    names = globals()

    for i in range(len(train_model_type_list)):
        names['features' + str(i)] = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
        names['labels' + str(i)] = tf.placeholder(tf.int64, [None, num_class])

    train_model_list = build_model()

    start_time = timer()
    for tidx, tm in enumerate(train_model_list):
        p = Process(target=train_model, args=(tm, train_batch_size_list[tidx], tidx,))
        p.start()
        p.join()
    end_time = timer()
    dur_time = end_time - start_time
    print("total training time(s): {}".format(dur_time))

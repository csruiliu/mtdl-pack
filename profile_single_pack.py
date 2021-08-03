import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
import itertools
from datetime import datetime
import os
import sys

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from tools.model_tool import ModelImporter
from tools.dataset_tool import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras
from tools.dataset_tool import load_mnist_image, load_mnist_label_onehot


def gen_confs():
    all_conf = [model_type_list, batch_size_list, opt_list, activation_list, learn_rate_list]
    hp_workload_conf = list(itertools.product(*all_conf))
    return hp_workload_conf


def profile_single_model(job):
    job_model_arch = job[0]
    job_model_type = job_model_arch.split('-')[0]
    job_num_layer = int(job_model_arch.split('-')[1])
    job_batch_size = job[1]
    job_opt = job[2]
    job_activation = job[3]
    job_learn_rate = job[4]

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize)

    model_name = '{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}'.format(net_instnace,
                                                          job_model_type,
                                                          job_num_layer,
                                                          job_batch_size,
                                                          job_learn_rate,
                                                          job_opt,
                                                          job_activation,
                                                          train_dataset)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dm = ModelImporter(job_model_type,
                       str(net_instnace),
                       job_num_layer,
                       img_height,
                       img_width,
                       num_channel,
                       num_class,
                       job_batch_size,
                       job_opt,
                       job_learn_rate,
                       job_activation,
                       batch_padding=True)

    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(features, is_training=True)
    train_step = model_entity.train(model_logit, labels)

    step_time = 0
    step_count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // job_batch_size
        for i in range(num_batch):
            print('step %d / %d' % (i + 1, num_batch))

            if i != 0:
                start_time = timer()

            batch_offset = i * job_batch_size
            batch_end = (i + 1) * job_batch_size

            if train_dataset == 'imagenet':
                batch_list = image_list[batch_offset:batch_end]
                train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
            else:
                train_feature_batch = train_feature[batch_offset:batch_end]

            train_label_batch = train_label[batch_offset:batch_end]

            sess.run(train_step, feed_dict={features: train_feature_batch, labels: train_label_batch})

            if i != 0:
                end_time = timer()
                dur_time = end_time - start_time
                print("step time:", dur_time)
                step_time += dur_time
                step_count += 1

    avg_step_time = step_time / step_count * 1000
    print('Job {}: {}'.format(model_name, avg_step_time))


def profile_pack_model(job_a, job_b):
    job_model_arch_a = job_a[0]
    job_model_type_a = job_model_arch_a.split('-')[0]
    job_num_layer_a = int(job_model_arch_a.split('-')[1])
    job_batch_size_a = job_a[1]
    job_opt_a = job_a[2]
    job_activation_a = job_a[3]
    job_learn_rate_a = job_a[4]

    model_name_a = '{0}-{1}-{2}-{3}-{4}-{5}-{6}'.format(job_model_type_a,
                                                        job_num_layer_a,
                                                        job_batch_size_a,
                                                        job_learn_rate_a,
                                                        job_opt_a,
                                                        job_activation_a,
                                                        train_dataset)

    job_model_arch_b = job_b[0]
    job_model_type_b = job_model_arch_b.split('-')[0]
    job_num_layer_b = int(job_model_arch_b.split('-')[1])
    job_batch_size_b = job_b[1]
    job_opt_b = job_b[2]
    job_activation_b = job_b[3]
    job_learn_rate_b = job_b[4]

    model_name_b = '{0}-{1}-{2}-{3}-{4}-{5}-{6}'.format(job_model_type_b,
                                                        job_num_layer_b,
                                                        job_batch_size_b,
                                                        job_learn_rate_b,
                                                        job_opt_b,
                                                        job_activation_b,
                                                        train_dataset)

    max_batch_size = max(job_batch_size_a, job_batch_size_b)

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize, size=2)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dm_a = ModelImporter(job_model_type_a,
                         str(net_instnace[0]),
                         job_num_layer_a,
                         img_height,
                         img_width,
                         num_channel,
                         num_class,
                         job_batch_size_a,
                         job_opt_a,
                         job_learn_rate_a,
                         job_activation_a,
                         batch_padding=True)
    model_entity_a = dm_a.get_model_entity()
    model_logit_a = model_entity_a.build(features, is_training=True)
    train_step_a = model_entity_a.train(model_logit_a, labels)

    dm_b = ModelImporter(job_model_type_b,
                         str(net_instnace[1]),
                         job_num_layer_b,
                         img_height,
                         img_width,
                         num_channel,
                         num_class,
                         job_batch_size_b,
                         job_opt_b,
                         job_learn_rate_b,
                         job_activation_b,
                         batch_padding=True)
    model_entity_b = dm_b.get_model_entity()
    model_logit_b = model_entity_b.build(features, is_training=True)
    train_step_b = model_entity_b.train(model_logit_b, labels)

    step_time = 0
    step_count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // max_batch_size
        for i in range(num_batch):
            print('step %d / %d' % (i + 1, num_batch))

            if i != 0:
                start_time = timer()

            batch_offset = i * max_batch_size
            batch_end = (i + 1) * max_batch_size

            if train_dataset == 'imagenet':
                batch_list = image_list[batch_offset:batch_end]
                train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
            else:
                train_feature_batch = train_feature[batch_offset:batch_end]

            train_label_batch = train_label[batch_offset:batch_end]

            sess.run([train_step_a, train_step_b], feed_dict={features: train_feature_batch, labels: train_label_batch})

            if i != 0:
                end_time = timer()
                dur_time = end_time - start_time
                print("step time:", dur_time)
                step_time += dur_time
                step_count += 1

    avg_step_time = step_time / step_count * 1000
    print(f'Pack {model_name_a} and {model_name_b}: {avg_step_time}')


if __name__ == "__main__":
    ##########################################
    # Hyperparameters
    ##########################################

    rand_seed_hyperband = 10000

    train_dataset = 'cifar10'
    model_type_list = ['densenet-121', 'resnet-50', 'mobilenet-1', 'mlp-1']
    batch_size_list = [32, 50, 64, 100]
    opt_list = ['Adam', 'SGD', 'Adagrad', 'Momentum']
    activation_list = ['sigmoid', 'leaky_relu', 'tanh', 'relu']
    learn_rate_list = [0.01, 0.001, 0.0001, 0.00001]

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
        test_img_path = cfg_path.mnist_eval_10k_img_path
        test_label_path = cfg_path.mnist_eval_10k_label_path

        train_feature = load_mnist_image(train_img_path)
        train_label = load_mnist_image(test_img_path)
        test_feature = load_mnist_image(test_img_path)
        test_label = load_mnist_label_onehot(test_label_path)

    hyperband_conf_list = gen_confs()

    ############################################
    # profile single model
    ############################################

    for job_conf in hyperband_conf_list:
        profile_single_model(job_conf)

    ############################################
    # profile pack model
    ############################################

    for job_conf_a in hyperband_conf_list:
        for job_conf_b in hyperband_conf_list:
            profile_pack_model(job_conf_a, job_conf_b)

import tensorflow as tf
from tensorflow.python.client import timeline
from multiprocessing import Pool
from timeit import default_timer as timer
import os
import sys
sys.path.append(os.path.abspath(".."))

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from models.model_importer import ModelImporter
from utils.utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot


def train_parallel(para_list_arg):
    model_type_arg = para_list_arg[0]
    model_id_arg = para_list_arg[1]
    num_layer_arg = para_list_arg[2]
    activation_arg = para_list_arg[3]
    batch_size_arg = para_list_arg[4]
    learn_rate_arg = para_list_arg[5]
    opt_arg = para_list_arg[6]
    num_epoch_arg = para_list_arg[7]
    train_dataset_arg = para_list_arg[8]

    model_name = '{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}'.format(model_id_arg, model_type_arg, num_layer_arg, batch_size_arg,
                                                          learn_rate_arg, opt_arg, num_epoch_arg, train_dataset_arg)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dm = ModelImporter(model_type_arg, str(model_id_arg), num_layer_arg, img_height, img_width, num_channel, num_class,
                       batch_size_arg, opt_arg, learn_rate_arg, activation_arg, batch_padding=False)

    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(features, is_training=True)
    train_op = model_entity.train(model_logit, labels)

    step_time = 0
    step_count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset_arg == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // batch_size_arg

        for e in range(num_epoch_arg):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                if i != 0:
                    start_time = timer()

                batch_offset = i * batch_size_arg
                batch_end = (i + 1) * batch_size_arg
                if train_dataset_arg == 'imagenet':
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
                    trace_file = open(profile_path + '/' + str(train_model_type_list) + '-'
                                      + str(batch_size_arg) + '-' + str(i) + '.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                else:
                    sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

                if i != 0:
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

    step_time_result = 'average step time (ms) of {}:{}'.format(model_name, step_time / step_count * 1000)
    return step_time_result


if __name__ == '__main__':

    ##########################################
    # Hyperparameters read from config
    ##########################################

    rand_seed = cfg_para.multi_rand_seed

    train_model_type_list = cfg_para.multi_model_type
    train_optimizer_list = cfg_para.multi_opt
    train_layer_num_list = cfg_para.multi_num_layer
    train_activation_list = cfg_para.multi_activation
    train_batch_size_list = cfg_para.multi_batch_size
    train_learn_rate_list = cfg_para.multi_learning_rate

    num_epoch = cfg_para.multi_num_epoch
    train_dataset = cfg_para.multi_train_dataset
    use_tf_timeline = cfg_para.multi_use_tb_timeline
    use_cpu = cfg_para.multi_use_cpu

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

    #####################################################
    # Build and train models in parallel
    #####################################################

    pool = Pool(processes=len(train_model_type_list))
    proc_para_list = list()
    for tidx in range(len(train_model_type_list)):
        para_list = list()
        para_list.append(train_model_type_list[tidx])
        para_list.append(tidx)
        para_list.append(train_layer_num_list[tidx])
        para_list.append(train_activation_list[tidx])
        para_list.append(train_batch_size_list[tidx])
        para_list.append(train_learn_rate_list[tidx])
        para_list.append(train_optimizer_list[tidx])
        para_list.append(num_epoch)
        para_list.append(train_dataset)

        proc_para_list.append(para_list)

    results = pool.map_async(train_parallel, proc_para_list)
    results_list = results.get()

    for rvalue in results_list:
        print(rvalue)
        
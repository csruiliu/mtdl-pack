import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from timeit import default_timer as timer
import os

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from tools.model_tool import ModelImporter
from tools.dataset_tool import load_dataset_para, load_train_dataset, load_imagenet_raw


def train_pack_diff_input():
    print('start training pack with different input')

    rand_seed_pack = cfg_para.multi_rand_seed

    model_type_list = cfg_para.multi_model_type
    optimizer_list = cfg_para.multi_opt
    num_layer_list = cfg_para.multi_num_layer
    activation_list = cfg_para.multi_activation
    batch_size_list = cfg_para.multi_batch_size
    learning_rate_list = cfg_para.multi_learning_rate

    num_epoch = cfg_para.multi_num_epoch
    train_dataset = cfg_para.multi_train_dataset
    use_tf_timeline = cfg_para.single_use_tb_timeline

    max_batch_size = max(batch_size_list)
    num_train_model = len(model_type_list)

    #################################################
    # load dataset
    #################################################

    img_width, img_height, num_channel, num_class = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)

    ##################################################
    # build packed model with different input
    ##################################################

    names = locals()
    for midx in range(num_train_model):
        names['features' + str(midx)] = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
        names['labels' + str(midx)] = tf.placeholder(tf.int64, [None, num_class])

    model_name_abbr = np.random.choice(rand_seed_pack, len(model_type_list), replace=False).tolist()
    train_op_pack = list()

    for midx, mt in enumerate(model_type_list):
        dm = ModelImporter(mt, str(model_name_abbr.pop()), num_layer_list[midx],
                           img_height, img_width, num_channel, num_class,
                           batch_size_list[midx], optimizer_list[midx],
                           learning_rate_list[midx], activation_list[midx], False)

        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(names['features' + str(midx)], is_training=True)
        train_op = model_entity.train(model_logit, names['labels' + str(midx)])

        train_op_pack.append(train_op)

    ##################################################
    # train packed model with different input
    ##################################################

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    step_time = 0
    step_count = 0

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    input_dict = dict()

    overall_time_start = timer()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label_input.shape[0] // max_batch_size

        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                if i != 0:
                    start_time = timer()

                for ridx in range(num_train_model):
                    batch_offset = i * max_batch_size
                    batch_end = (i + 1) * max_batch_size
                    batch_list = image_list[batch_offset:batch_end]
                    names['train_feature_batch' + str(ridx)] = load_imagenet_raw(train_feature_input, batch_list,
                                                                                 img_height, img_width)
                    names['train_label_batch' + str(ridx)] = train_feature_input[batch_offset:batch_end, :]
                    input_dict[names['features' + str(ridx)]] = names['train_feature_batch' + str(ridx)]
                    input_dict[names['labels' + str(ridx)]] = names['train_label_batch' + str(ridx)]

                if use_tf_timeline:
                    profile_path = cfg_path.profile_path
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    sess.run(train_op_pack, feed_dict=input_dict, options=run_options, run_metadata=run_metadata)

                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(profile_path + '/' + '-'.join(map(str, set(model_type_list))) + '-' +
                                      str(len(model_type_list)) + '-' +
                                      '-'.join(map(str, set(batch_size_list))) +
                                      '-' + str(i) + '.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                else:
                    sess.run(train_op_pack, feed_dict=input_dict)

                if i != 0:
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

    overall_time_end = timer()
    overall_time = overall_time_end - overall_time_start
    print('overall training time (s):{}, average step time (ms):{}'
          .format(overall_time, step_time / step_count * 1000))


def train_pack():
    print('start training pack')

    rand_seed_pack = cfg_para.multi_rand_seed

    model_type_list = cfg_para.multi_model_type
    optimizer_list = cfg_para.multi_opt
    num_layer_list = cfg_para.multi_num_layer
    activation_list = cfg_para.multi_activation
    batch_size_list = cfg_para.multi_batch_size
    learning_rate_list = cfg_para.multi_learning_rate

    if len(set(batch_size_list)) == 1:
        is_batch_padding = False
    else:
        is_batch_padding = True

    num_epoch = cfg_para.multi_num_epoch
    train_dataset = cfg_para.multi_train_dataset
    use_tf_timeline = cfg_para.single_use_tb_timeline

    max_batch_size = max(batch_size_list)

    #################################################
    # load dataset
    #################################################

    img_width, img_height, num_channel, num_class = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)

    #########################
    # build packed model
    #########################

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    model_name_abbr = np.random.choice(rand_seed_pack, len(model_type_list), replace=False).tolist()
    train_op_pack = list()

    for midx, mt in enumerate(model_type_list):
        dm = ModelImporter(mt, str(model_name_abbr.pop()), num_layer_list[midx], img_height,
                           img_width, num_channel, num_class, batch_size_list[midx],
                           optimizer_list[midx], learning_rate_list[midx], activation_list[midx],
                           batch_padding=is_batch_padding)

        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        train_op_pack.append(train_op)

    #########################
    # train packed model
    #########################

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    step_time = 0
    step_count = 0

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    overall_time_start = timer()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label_input.shape[0] // max_batch_size

        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                if i != 0:
                    start_time = timer()

                batch_offset = i * max_batch_size
                batch_end = (i + 1) * max_batch_size
                if train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_feature_input, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature_input[batch_offset:batch_end]

                train_label_batch = train_label_input[batch_offset:batch_end]

                if use_tf_timeline:
                    profile_path = cfg_path.profile_path
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    sess.run(train_op_pack, feed_dict={features: train_feature_batch, labels: train_label_batch},
                             options=run_options, run_metadata=run_metadata)

                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(profile_path + '/' + '-'.join(map(str, set(model_type_list))) + '-' +
                                      str(len(model_type_list)) + '-'.join(map(str, set(batch_size_list))) +
                                      '-' + str(i) + '.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                else:
                    sess.run(train_op_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})

                if i != 0:
                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

    overall_time_end = timer()
    overall_time = overall_time_end - overall_time_start
    print('overall training time (s):{}, average step time (ms):{}'
          .format(overall_time, step_time / step_count * 1000))

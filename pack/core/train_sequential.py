import tensorflow as tf
from tensorflow.python.client import timeline
import os
from multiprocessing import Process
import numpy as np
from timeit import default_timer as timer

import pack.config.config_parameter as cfg_para
import pack.config.config_path as cfg_path
from pack.core.dataset_loader import load_dataset_para, load_train_dataset
from pack.core.model_importer import ModelImporter
from pack.tools.img_tool import load_imagenet_raw


def train_model(train_step_arg, batch_size_arg, model_type_arg, tidx_arg, global_args):
    train_dataset = cfg_para.multi_train_dataset
    num_epoch = cfg_para.multi_num_epoch
    use_tf_timeline = cfg_para.multi_use_tb_timeline
    use_cpu = cfg_para.multi_use_cpu

    if use_cpu:
        train_device = '/cpu:0'
    else:
        train_device = '/gpu:0'

    img_width, img_height, num_channel, num_class = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    with tf.device(train_device):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label_input.shape[0] // batch_size_arg

            for e in range(num_epoch):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                    batch_offset = i * batch_size_arg
                    batch_end = (i + 1) * batch_size_arg
                    if train_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        feature_batch = load_imagenet_raw(train_feature_input, batch_list, img_height, img_width)
                    else:
                        feature_batch = train_feature_input[batch_offset:batch_end]

                    label_batch = train_label_input[batch_offset:batch_end]

                    if use_tf_timeline:
                        profile_path = cfg_path.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(train_step_arg, feed_dict={global_args['features' + str(tidx_arg)]: feature_batch,
                                                            global_args['labels' + str(tidx_arg)]: label_batch},
                                 options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(model_type_arg) + '-'
                                          + str(batch_size_arg) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(train_step_arg, feed_dict={global_args['features' + str(tidx_arg)]: feature_batch,
                                                            global_args['labels' + str(tidx_arg)]: label_batch})


def train_sequential():
    print('start training sequential')

    rand_seed = cfg_para.multi_rand_seed

    model_type_list = cfg_para.multi_model_type
    optimizer_list = cfg_para.multi_opt
    num_layer_list = cfg_para.multi_num_layer
    activation_list = cfg_para.multi_activation
    batch_size_list = cfg_para.multi_batch_size
    learning_rate_list = cfg_para.multi_learning_rate

    train_dataset = cfg_para.multi_train_dataset

    ##########################################
    # load dataset parameters
    ##########################################

    img_width, img_height, num_channel, num_class = load_dataset_para(train_dataset)

    ##########################################
    # build models
    ##########################################

    names = globals()
    for idx in range(len(model_type_list)):
        names['features' + str(idx)] = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
        names['labels' + str(idx)] = tf.placeholder(tf.int64, [None, num_class])

    train_op_list = list()
    model_name_abbr = np.random.choice(rand_seed, len(model_type_list), replace=False).tolist()
    for midx, mvalue in enumerate(model_type_list):
        dm = ModelImporter(mvalue, str(model_name_abbr.pop()), num_layer_list[midx],
                           img_width, img_height, num_channel, num_class,
                           batch_size_list[midx], optimizer_list[midx],
                           learning_rate_list[midx], activation_list[midx], batch_padding=False)

        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(names['features' + str(midx)], is_training=True)
        train_op = model_entity.train(model_logit, names['labels' + str(midx)])
        train_op_list.append(train_op)

    #########################
    # train models
    #########################

    start_time = timer()
    for tidx, tm in enumerate(train_op_list):
        p = Process(target=train_model, args=(tm, batch_size_list[tidx],
                                              model_type_list[tidx], tidx, names))
        p.start()
        p.join()
    end_time = timer()
    dur_time = end_time - start_time
    print("total training time(s): {}".format(dur_time))

import tensorflow as tf
from tensorflow.python.client import timeline
from multiprocessing import Pool
from timeit import default_timer as timer
import os

import pack.config.config_parameter as cfg_para
import pack.config.config_path as cfg_path

from pack.core.dataset_loader import data_loader
from pack.core.model_importer import ModelImporter
from pack.tools.img_tool import load_imagenet_raw


def train_model(job_id):

    model_type_list = cfg_para.multi_model_type
    num_layer_list = cfg_para.multi_num_layer
    activation_list = cfg_para.multi_activation
    batch_size_list = cfg_para.multi_batch_size
    learning_rate_list = cfg_para.multi_learning_rate
    optimizer_list = cfg_para.multi_opt

    model_type = model_type_list[job_id]
    num_layer = num_layer_list[job_id]
    activation = activation_list[job_id]
    batch_size = batch_size_list[job_id]
    learning_rate = learning_rate_list[job_id]
    optimizer = optimizer_list[job_id]

    num_epoch = cfg_para.multi_num_epoch
    train_dataset = cfg_para.multi_train_dataset
    use_tf_timeline = cfg_para.multi_use_tb_timeline
    use_cpu = cfg_para.multi_use_cpu

    if use_cpu:
        train_device = '/cpu:0'
    else:
        train_device = '/gpu:0'

    model_name = '{0}-{1}-{2}-{3}-{4}-{5}-{6}-{7}'.format(job_id, model_type, num_layer,
                                                          batch_size, learning_rate, optimizer,
                                                          num_epoch, train_dataset)

    ##########################################
    # load dataset
    ##########################################

    args_list = data_loader(train_dataset)

    img_width = args_list[0]
    img_height = args_list[1]
    num_channel = args_list[2]
    num_class = args_list[3]
    train_feature_input = args_list[4]
    train_label_input = args_list[5]

    ##########################################
    # build model
    ##########################################

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dm = ModelImporter(model_type, str(job_id), num_layer, img_height,
                       img_width, num_channel, num_class, batch_size,
                       optimizer, learning_rate, activation, batch_padding=False)

    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(features, is_training=True)
    train_op = model_entity.train(model_logit, labels)

    ##########################################
    # train model
    ##########################################

    step_time = 0
    step_count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    with tf.device(train_device):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label_input.shape[0] // batch_size

            for e in range(num_epoch):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))

                    if i != 0:
                        start_time = timer()

                    batch_offset = i * batch_size
                    batch_end = (i + 1) * batch_size
                    if train_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input, batch_list,
                                                                img_height, img_width)
                    else:
                        train_feature_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]

                    if use_tf_timeline:
                        profile_path = cfg_path.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch},
                                 options=run_options, run_metadata=run_metadata)

                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(model_type) + '-'
                                          + str(batch_size) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

                    if i != 0:
                        end_time = timer()
                        dur_time = end_time - start_time
                        print("step time:", dur_time)
                        step_time += dur_time
                        step_count += 1

    step_time_result = 'average step time (ms) of {}: {}'.format(model_name, step_time / step_count * 1000)
    return step_time_result


def train_parallel():
    print('start training parallel')

    model_type_list = cfg_para.multi_model_type

    #####################################################
    # train models in parallel
    #####################################################

    pool = Pool(processes=len(model_type_list))
    proc_para_list = list(range(len(model_type_list)))

    overall_start_time = timer()
    results = pool.map_async(train_parallel, proc_para_list)
    results_list = results.get()
    overall_end_time = timer()
    overall_dur_time = overall_end_time - overall_start_time

    for rvalue in results_list:
        print(rvalue)

    print('Overall parallel training time(s): {}'.format(overall_dur_time))

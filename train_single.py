import numpy as np
from timeit import default_timer as timer
import os
import tensorflow as tf
from tensorflow.python.client import timeline

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from tools.model_tool import ModelImporter
from tools.dataset_tool import load_dataset_para, load_train_dataset, load_eval_dataset, load_imagenet_raw


def train_single():
    print('start training single')
    rand_seed = cfg_para.single_rand_seed
    num_epoch = cfg_para.single_num_epoch

    model_type = cfg_para.single_model_type
    num_layer = cfg_para.single_num_layer
    learning_rate = cfg_para.single_learning_rate
    activation = cfg_para.single_activation
    batch_size = cfg_para.single_batch_size
    optimizer = cfg_para.single_opt

    train_dataset = cfg_para.single_train_dataset
    use_tf_timeline = cfg_para.single_use_tb_timeline
    use_cpu = cfg_para.single_use_cpu

    if use_cpu:
        train_device = '/cpu:0'
    else:
        train_device = '/gpu:0'

    ##########################################
    # load dataset
    ##########################################

    img_width, img_height, num_channel, num_class = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    ##########################################
    # build model
    ##########################################

    feature_ph = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    label_ph = tf.placeholder(tf.int64, [None, num_class])

    model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()

    dm = ModelImporter(model_type,
                       str(model_name_abbr.pop()),
                       num_layer,
                       img_height,
                       img_width,
                       num_channel,
                       num_class,
                       batch_size,
                       optimizer,
                       learning_rate,
                       activation,
                       batch_padding=False)

    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(feature_ph, is_training=True)
    train_op = model_entity.train(model_logit, label_ph)
    eval_op = model_entity.evaluate(model_logit, label_ph)

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

    overall_time_start = timer()
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
                        sess.run(train_op,
                                 feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch},
                                 options=run_options,
                                 run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + str(model_type) + '-' + str(batch_size) + '-'
                                          + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(train_op, feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch})

                    if i != 0:
                        end_time = timer()
                        dur_time = end_time - start_time
                        print("step time:", dur_time)
                        step_time += dur_time
                        step_count += 1

            acc_avg = sess.run(eval_op, feed_dict={feature_ph: eval_feature_input,
                                                   label_ph: eval_label_input})

    print('evaluation accuracy:{}'.format(acc_avg))

    overall_time_end = timer()
    overall_time = overall_time_end - overall_time_start

    print(f'overall training time (s):{overall_time}, average step time (ms):{step_time / step_count * 1000}')


if __name__ == '__main__':
    train_single()
import tensorflow as tf
from tensorflow.python.client import timeline
from timeit import default_timer as timer
import os

from pack.common.model_builder import build_model_single
from pack.common.dataset_loader import data_loader
from pack.tools.img_tool import load_imagenet_raw
import pack.config.config_parameter as cfg_para
import pack.config.config_path as cfg_path


def train_model():
    step_time = 0
    step_count = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature))

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
                    train_feature_batch = load_imagenet_raw(train_feature, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature[batch_offset:batch_end]

                train_label_batch = train_label[batch_offset:batch_end]

                if use_tf_timeline:
                    profile_path = cfg_path.profile_path
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    sess.run(train_op, feed_dict={feature_ph: train_feature_batch, label_ph: train_label_batch},
                             options=run_options, run_metadata=run_metadata)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(profile_path + '/' + str(train_model_type) + '-' + str(train_batch_size) + '-'
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

        acc_avg = sess.run(eval_op, feed_dict={feature_ph: test_feature, label_ph: test_label})

    print('evaluation accuracy:{}'.format(acc_avg))

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

    args_list = data_loader(train_dataset)

    img_width = args_list[0]
    img_height = args_list[1]
    num_channel = args_list[2]
    num_class = args_list[3]
    train_feature = args_list[4]
    train_label = args_list[5]
    test_feature = args_list[6]
    test_label = args_list[7]

    ###########################################################
    # Build and train model due to input dataset
    ###########################################################

    feature_ph = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    label_ph = tf.placeholder(tf.int64, [None, num_class])

    train_op, eval_op = build_model_single(rand_seed, train_model_type, feature_ph, label_ph,
                                           train_layer_num, img_height, img_width, num_channel,
                                           num_class, train_batch_size, train_optimizer,
                                           train_learn_rate, train_activation, False)

    train_model()

from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import numpy as np
from timeit import default_timer as timer

import config as cfg_yml
from dnn_model import DnnModel
from img_utils import *


def buildModels():
    model_name_abbr = np.random.choice(randSeed, len(trainModel), replace=False).tolist()
    trainOpCollection = list()

    for idx, mt in enumerate(trainModel):
        dm = DnnModel(mt, str(model_name_abbr.pop()), trainNumLayer[idx], imgHeight, imgWidth, numChannels, numClasses,
                      trainBatchSize[idx], trainOptimizer[idx], trainLearnRate[idx], trainActivation[idx], batchPadding)
        modelEntity = dm.getModelEntity()

        if sameInput:
            modelLogit = modelEntity.build(features)
            trainStep = modelEntity.train(modelLogit, labels)
        else:
            modelLogit = modelEntity.build(names['features' + str(idx)])
            trainStep = modelEntity.train(modelLogit, names['labels' + str(idx)])

        trainOpCollection.append(trainStep)

    return trainOpCollection

def profileStepRawImageSameInput():
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    image_list = sorted(os.listdir(image_path_raw))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize

        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                if (i + 1) % recordMarker == 0:
                    start_time = timer()
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]

                    if useTimeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed},
                                 options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + '-'.join(map(str, set(trainModel))) + '-' + str(len(trainModel)) + '-' + '-'.join(map(str, set(trainBatchSize))) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

                else:
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                    sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

    print("average step time (ms):", step_time / step_count * 1000)

def profileStepRawImageDiffInput():
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(image_path_raw))
    input_model_num = len(trainOpPack)
    input_dict = dict()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))

                if (i + 1) % recordMarker == 0:
                    start_time = timer()
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        batch_list = image_list[batch_offset:batch_end]
                        names['X_mini_batch_feed' + str(ridx)] = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                        names['Y_mini_batch_feed' + str(ridx)] = Y_data[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]

                    if useTimeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOpPack, feed_dict=input_dict, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + '-'.join(map(str, set(trainModel))) + '-' + str(len(trainModel)) + '-' + '-'.join(map(str, set(trainBatchSize))) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOpPack, feed_dict=input_dict)

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1
                else:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        batch_list = image_list[batch_offset:batch_end]
                        names['X_mini_batch_feed' + str(ridx)] = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                        names['Y_mini_batch_feed' + str(ridx)] = Y_data[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                    sess.run(trainOpPack, feed_dict=input_dict)

    print("average step time (ms):", step_time / step_count * 1000)


def profileEpochRawImageSameInput():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(image_path_raw))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize

        start_time = timer()
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))

                start_time = timer()
                batch_offset = i * maxBatchSize
                batch_end = (i + 1) * maxBatchSize
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(numEpochs, dur_time))


def profileEpochRawImageDiffInput():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(image_path_raw))
    input_model_num = len(trainOpPack)
    input_dict = dict()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        start_time = timer()
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                for ridx in range(input_model_num):
                    rand_idx = int(np.random.choice(num_batch_list, 1))
                    batch_offset = rand_idx * maxBatchSize
                    batch_end = (rand_idx + 1) * maxBatchSize
                    batch_list = image_list[batch_offset:batch_end]
                    names['X_mini_batch_feed' + str(ridx)] = load_imagenet_raw(image_path_raw, batch_list, imgHeight, imgWidth)
                    names['Y_mini_batch_feed' + str(ridx)] = Y_data[batch_offset:batch_end, :]
                    input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                    input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]
                sess.run(trainOpPack, feed_dict=input_dict)

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(numEpochs, dur_time))


def profileStepSameInput():
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize

        for e in range(numEpochs):
            for i in range(num_batch):
                if (i + 1) % recordMarker == 0:
                    start_time = timer()
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]

                    if useTimeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed}, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + '-'.join(map(str, set(trainModel))) + '-' + str(len(trainModel)) + '-' + '-'.join(map(str, set(trainBatchSize))) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

                else:
                    batch_offset = i * maxBatchSize
                    batch_end = (i + 1) * maxBatchSize
                    X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                    Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                    sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

    print("average step time (ms):", step_time / step_count * 1000)
    

def profileStepDiffInput():
    step_time = 0
    step_count = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    input_model_num = len(trainOpPack)
    input_dict = dict()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        for e in range(numEpochs):
            for i in range(num_batch):
                if (i + 1) % recordMarker == 0:
                    start_time = timer()
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        names['X_mini_batch_feed' + str(ridx)] = X_data[batch_offset:batch_end, :, :, :]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_data[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]

                    if useTimeline:
                        profile_path = cfg_yml.profile_path
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        sess.run(trainOpPack, feed_dict=input_dict, options=run_options, run_metadata=run_metadata)
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        trace_file = open(profile_path + '/' + '-'.join(map(str, set(trainModel))) + '-' + str(len(trainModel)) + '-' + '-'.join(map(str, set(trainBatchSize))) + '-' + str(i) + '.json', 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                    else:
                        sess.run(trainOpPack, feed_dict=input_dict)

                    end_time = timer()
                    dur_time = end_time - start_time
                    print("step time:", dur_time)
                    step_time += dur_time
                    step_count += 1

                else:
                    for ridx in range(input_model_num):
                        rand_idx = int(np.random.choice(num_batch_list, 1))
                        batch_offset = rand_idx * maxBatchSize
                        batch_end = (rand_idx + 1) * maxBatchSize
                        names['X_mini_batch_feed' + str(ridx)] = X_data[batch_offset:batch_end, :, :, :]
                        names['Y_mini_batch_feed' + str(ridx)] = Y_data[batch_offset:batch_end, :]
                        input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                        input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]

                    sess.run(trainOpPack, feed_dict=input_dict)
    
    print("average step time (ms):", step_time / step_count * 1000)

def profileEpochSameInput():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize
        start_time = timer()
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                batch_offset = i * maxBatchSize
                batch_end = (i + 1) * maxBatchSize
                X_mini_batch_feed = X_data[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end, :]
                sess.run(trainOpPack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(numEpochs, dur_time))


def profileEpochDiffInput():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    input_model_num = len(trainOpPack)
    input_dict = dict()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize
        num_batch_list = np.arange(num_batch)

        start_time = timer()
        for e in range(numEpochs):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, numEpochs, i + 1, num_batch))
                for ridx in range(input_model_num):
                    rand_idx = int(np.random.choice(num_batch_list, 1))
                    batch_offset = rand_idx * maxBatchSize
                    batch_end = (rand_idx + 1) * maxBatchSize
                    names['X_mini_batch_feed' + str(ridx)] = X_data[batch_offset:batch_end, :, :, :]
                    names['Y_mini_batch_feed' + str(ridx)] = Y_data[batch_offset:batch_end, :]
                    input_dict[names['features' + str(ridx)]] = names['X_mini_batch_feed' + str(ridx)]
                    input_dict[names['labels' + str(ridx)]] = names['Y_mini_batch_feed' + str(ridx)]

                sess.run(trainOpPack, feed_dict=input_dict)
        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(numEpochs, dur_time))



if __name__ == '__main__':

    #########################
    # Parameters
    #########################

    imgWidth = cfg_yml.pack_img_width
    imgHeight = cfg_yml.pack_img_height
    numChannels = cfg_yml.pack_num_channels
    numClasses = cfg_yml.pack_num_classes
    numEpochs = cfg_yml.pack_num_epoch
    trainModel = cfg_yml.pack_model_type
    trainBatchSize = cfg_yml.pack_batch_size
    trainOptimizer = cfg_yml.pack_opt
    randSeed = cfg_yml.pack_rand_seed
    trainNumLayer = cfg_yml.pack_num_layer
    trainLearnRate = cfg_yml.pack_learning_rate
    trainActivation = cfg_yml.pack_activation

    batchPadding = cfg_yml.pack_batch_padding
    useRawImage = cfg_yml.pack_use_raw_image
    measureStep = cfg_yml.pack_measure_step
    useCPU = cfg_yml.pack_use_cpu
    sameInput = cfg_yml.pack_same_input
    recordMarker = cfg_yml.pack_record_marker
    useTimeline = cfg_yml.pack_use_tb_timeline

    maxBatchSize = max(trainBatchSize)
    numTrainModel = len(trainModel)

    if useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #########################
    # Build
    #########################

    if sameInput:
        features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        labels = tf.placeholder(tf.int64, [None, numClasses])

    else:
        names = locals()
        for midx in range(numTrainModel):
            names['features' + str(midx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
            names['labels' + str(midx)] = tf.placeholder(tf.int64, [None, numClasses])

    trainOpPack = buildModels()

    #########################
    # Train
    #########################

    image_path_raw = cfg_yml.imagenet_t1k_img_path
    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    label_path = cfg_yml.imagenet_t1k_label_path

    if useRawImage:
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            if sameInput:
                profileStepRawImageSameInput()
            else:
                profileStepRawImageDiffInput()
        else:
            if sameInput:
                profileEpochRawImageSameInput()
            else:
                profileEpochRawImageDiffInput()
    else:
        X_data = load_imagenet_bin(image_path_bin, numChannels, imgWidth, imgHeight)
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        if measureStep:
            if sameInput:
                profileStepSameInput()
            else:
                profileStepDiffInput()
        else:
            if sameInput:
                profileEpochSameInput()
            else:
                profileEpochDiffInput()

from __future__ import division
import tensorflow as tf
from tensorflow.python.client import timeline
import os
from multiprocessing import Process
import numpy as np
from timeit import default_timer as timer

import config as cfg_yml
from dnn_model import DnnModel
from img_utils import *

def buildModels(train_model, num_layer, input_w, input_h, num_channels, num_classes, batch_size, optimizer,
                learning_rate, activation, rand_seed):

    trainCollection = []
    names = locals()

    model_name_abbr = np.random.choice(rand_seed, len(trainModel), replace=False).tolist()
    for tidx, mt in enumerate(train_model):
        names['features' + str(tidx)] = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
        names['labels' + str(tidx)] = tf.placeholder(tf.int64, [None, numClasses])

        dm = DnnModel(mt, str(model_name_abbr.pop()), num_layer[tidx], input_w, input_h, num_channels, num_classes,
                      batch_size[tidx], optimizer[tidx], learning_rate[tidx], activation[tidx], False)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(names['features' + str(tidx)])
        trainStep = modelEntity.train(modelLogit, names['labels' + str(tidx)])
        trainCollection.append(trainStep)

    return trainCollection

def profileEpochRawImage(trainStep, num_epoch, batch_size, X_train_path, Y_train, tidx):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(X_train_path))
    names = locals()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        start_time = timer()
        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i + 1) * trainBatchSize
                batch_list = image_list[batch_offset:batch_end]
                X_mini_batch_feed = load_imagenet_raw(X_train_path, batch_list, imgHeight, imgWidth)
                Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]

                sess.run(trainStep, feed_dict={names['features' + str(tidx)]: X_mini_batch_feed,
                                               names['labels' + str(tidx)]: Y_mini_batch_feed})
        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))

def profileEpoch(trainStep, num_epoch, batch_size, X_train, Y_train, tidx):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    names = locals()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_train.shape[0] // trainBatchSize
        start_time = timer()

        for e in range(num_epoch):
            for i in range(num_batch):
                print('epoch %d / %d, step %d / %d' % (e + 1, num_epoch, i + 1, num_batch))
                batch_offset = i * trainBatchSize
                batch_end = (i + 1) * trainBatchSize
                X_mini_batch_feed = X_train[batch_offset:batch_end, :, :, :]
                Y_mini_batch_feed = Y_train[batch_offset:batch_end, :]
                sess.run(trainStep, feed_dict={names['features' + str(tidx)]: X_mini_batch_feed,
                                               names['labels' + str(tidx)]: Y_mini_batch_feed})

        end_time = timer()
        dur_time = end_time - start_time
        print("training time ({} epoch): {} seconds".format(num_epoch, dur_time))


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

    useRawImage = cfg_yml.pack_use_raw_image
    measureStep = cfg_yml.pack_measure_step
    useCPU = cfg_yml.pack_use_cpu
    recordMarker = cfg_yml.pack_record_marker
    useTimeline = cfg_yml.pack_use_tb_timeline

    maxBatchSize = max(trainBatchSize)

    if useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #########################
    # Build and Train
    #########################

    trainCollection = buildModels(trainModel, trainNumLayer, imgHeight, imgWidth, numChannels, numClasses, trainBatchSize,
                              trainOptimizer, trainLearnRate, trainActivation, randSeed)

    image_path_bin = cfg_yml.imagenet_t1k_bin_path
    image_path_raw = cfg_yml.imagenet_t1k_img_path
    label_path = cfg_yml.imagenet_t1k_label_path

    if useRawImage:
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        start_time = timer()
        for tidx, tc in enumerate(trainCollection):
            p = Process(target=profileEpochRawImage, args=(tc, numEpochs, trainBatchSize[tidx], image_path_raw, Y_data, tidx,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("total training time:", dur_time)

    else:
        X_data = load_imagenet_bin(image_path_bin, numChannels, imgWidth, imgHeight)
        Y_data = load_imagenet_labels_onehot(label_path, numClasses)

        start_time = timer()
        for tidx, tc in enumerate(trainCollectionn):
            p = Process(target=profileEpoch, args=(tc, numEpochs, trainBatchSize[tidx], X_data, Y_data, tidx,))
            p.start()
            print(p.pid)
            p.join()
        end_time = timer()
        dur_time = end_time - start_time
        print("total training time:", dur_time)

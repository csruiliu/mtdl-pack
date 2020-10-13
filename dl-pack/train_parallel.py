from __future__ import division

import tensorflow as tf
from multiprocessing import Pool
import argparse
from timeit import default_timer as timer

from img_utils import *
from dnn_model import DnnModel

def read_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--models', action='store', type=str, help='models need to train, model name split by comma, example: resnet,resnet')
    parser.add_argument('-b', '--batchsizes', action='store', type=str, help='batch size list, batch size split by comma, example: 32,32')
    parser.add_argument('-o', '--optimizers', action='store', type=str, help='optimizer list, opts split by comma, example: SGD,SGD')
    parser.add_argument('-e', '--epoch', action='store', type=str, help='indicate how many epoches will run for each model')
    
    parser.add_argument('-r', '--record', action='store', type=int, default=3, help='indicate the record interval for measurement')

    parser.add_argument('-p', '--preproc', action='store_true', default=False, help='use preproc to transform the data before training or not')
    parser.add_argument('-c', '--usecpu', action='store_true', default=False, help='use same compute and apply to update gradient or not')
    parser.add_argument('-t', '--epochmeasure', action='store_true', default=False, help='measure the training epoch or step')

    args = parser.parse_args()

    modelsTrain = args.models
    batchSizeTrain = args.batchsizes
    optTrain = args.optimizers
    epochTrain = args.epoch
    
    hasPreproc = args.preproc
    measureEpoch = args.epochmeasure
    useCPU = args.usecpu
    record = args.record
    
    modelsTrainList = modelsTrain.split(',')
    batchSizeTrainList = batchSizeTrain.split(',')
    optTrainList = optTrain.split(',')
    epochTrainList = epochTrain.split(',')

    proc_num = 0

    if len(modelsTrainList) == len(batchSizeTrainList) and len(modelsTrainList) == len(optTrainList) and len(modelsTrainList) == len(epochTrainList):
        proc_num = len(modelsTrainList)
    else:
        print("number of model, batch size and opt don't match")
    
    return proc_num, modelsTrainList, batchSizeTrainList, optTrainList, epochTrainList, record, hasPreproc, measureEpoch, useCPU


def execParallelPreproc(pls):
    model_name = pls[0]
    model_instance = pls[1]
    batch_size = pls[2]
    opt = pls[3]
    num_epoch = pls[4]
    X_train_path = pls[5]
    Y_train_path = pls[6]

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    
    dm = DnnModel(model_name, str(model_instance), 1, imgWidth, imgHeight, numClasses, batch_size, opt)
    
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    Y_train = load_labels_onehot(Y_train_path, numClasses)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    image_list = sorted(os.listdir(X_train_path))
    if measureTrainEpoch:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_train.shape[0] // batch_size
            for e in range(num_epoch):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    batch_list = image_list[batch_offset:batch_end]   
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
    else:
        step_time = 0
        step_count = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_train.shape[0] // batch_size
            for e in range(num_epoch):
                for i in range(num_batch):
                    start_time = timer()
                    print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))      
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    batch_list = image_list[batch_offset:batch_end]   
                    X_mini_batch_feed = load_image_dir(X_train_path, batch_list, imgHeight, imgWidth)
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    step_time += dur_time
                    step_count += 1
        avg_step_time = step_time / step_count * 1000
        return avg_step_time


def execParallel(pls):
    model_name = pls[0]
    model_instance = pls[1]
    batch_size = pls[2]
    opt = pls[3]
    num_epoch = pls[4]
    X_train_path = pls[5]
    Y_train_path = pls[6]

    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    
    dm = DnnModel(model_name, str(model_instance), 1, imgWidth, imgHeight, numClasses, batch_size, opt)
    
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    X_train = load_images_bin(X_train_path, numChannels, imgWidth, imgHeight)
    Y_train = load_labels_onehot(Y_train_path, numClasses)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if measureTrainEpoch:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_train.shape[0] // batch_size
            for e in range(num_epoch):
                for i in range(num_batch):
                    print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))      
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
    else:
        step_time = 0
        step_count = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = Y_train.shape[0] // batch_size
            for e in range(num_epoch):
                for i in range(num_batch):
                    start_time = timer()
                    print('epoch %d / %d, step %d / %d' %(e+1, num_epoch, i+1, num_batch))      
                    batch_offset = i * batch_size
                    batch_end = (i+1) * batch_size
                    X_mini_batch_feed = X_train[batch_offset:batch_end,:,:,:]
                    Y_mini_batch_feed = Y_train[batch_offset:batch_end,:]
                    sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                    end_time = timer()
                    dur_time = end_time - start_time
                    step_time += dur_time
                    step_count += 1
        avg_step_time = step_time / step_count * 1000
        return avg_step_time


if __name__ == '__main__':
    #image_dir = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k'
    image_dir = '/tank/local/ruiliu/dataset/imagenet10k'
    #image_dir = '/local/ruiliu/dataset/imagenet10k'

    #bin_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k.bin'
    bin_path = '/tank/local/ruiliu/dataset/imagenet10k.bin'
    #bin_path = '/local/ruiliu/dataset/imagenet10k.bin'

    #label_path = '/home/ruiliu/Development/mtml-tf/dataset/imagenet1k-label.txt'
    label_path = '/tank/local/ruiliu/dataset/imagenet10k-label.txt'
    #label_path = '/local/ruiliu/dataset/imagenet10k-label.txt'

    imgWidth = 224
    imgHeight = 224
    numClasses = 1000
    numChannels = 3
    
    numProc, modelsTrainList, batchSizeTrainList, optTrainList, epochTrainList, recordInterval, hasPreproc, measureTrainEpoch, useCPU = read_args()
    modelNameAbbr = np.random.choice(100000, numProc, replace=False).tolist()

    if useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    if measureTrainEpoch:
        start_time = timer()

    pool = Pool(processes=numProc)
    proc_list = []
    for pidx in range(numProc):
        para_list = []
        para_list.append(modelsTrainList[pidx])
        para_list.append(modelNameAbbr[pidx])
        para_list.append(int(batchSizeTrainList[pidx]))
        para_list.append(optTrainList[pidx])
        para_list.append(int(epochTrainList[pidx]))
        para_list.append(image_dir)
        para_list.append(label_path)   
        proc_list.append(para_list)
    
    if hasPreproc:
        results = pool.map_async(execParallelPreproc, proc_list)
    else:
        results = pool.map_async(execParallel, proc_list)

    results.wait()
    if results.ready():
        if results.successful():
            print(results.get())

    if measureTrainEpoch:
        end_time = timer()
        dur_time = end_time - start_time
        print("total running time:", dur_time)



import tensorflow as tf
import config as cfg_yml
import itertools
from datetime import datetime
import numpy as np
import sys
from multiprocessing import Process
from timeit import default_timer as timer
import argparse
import os

from img_utils import *
from dnn_model import DnnModel

imgWidth = cfg_yml.imgWidth
imgHeight = cfg_yml.imgHeight
numChannels = cfg_yml.numChannels
numClasses = cfg_yml.numClasses
rand_seed = cfg_yml.rand_seed
learning_rate = cfg_yml.learning_rate[0]

batch_size_global = cfg_yml.batch_size
opt_conf_global = cfg_yml.opt_conf
model_layer_global = cfg_yml.model_layer
activation_global = cfg_yml.activation
model_type_global = cfg_yml.model_type
cifar_10_path = cfg_yml.cifar_10_path

#imagenet_t1k_img_path = cfg_yml.imagenet_t1k_img_path
#imagenet_t1k_label_path = cfg_yml.imagenet_t1k_label_path

#imagenet_image_list = sorted(os.listdir(imagenet_t1k_img_path))
#Y_data = load_imagenet_labels_onehot(imagenet_t1k_label_path, numClasses)
X_data, Y_data = load_cifar_train(cifar_10_path)

def gen_confs():
    all_conf = [model_type_global, batch_size_global, opt_conf_global, model_layer_global, activation_global]
    hp_conf = list(itertools.product(*all_conf))
    return hp_conf

def single_eval(conf_model):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    model_type = conf_model[0]
    batch_size = conf_model[1]
    opt = conf_model[2]
    model_layer = conf_model[3]
    activation = conf_model[4]
    desire_steps = Y_data.shape[0] // batch_size

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize)

    dm = DnnModel(model_type, str(net_instnace), model_layer, imgWidth, imgHeight, numChannels, numClasses, batch_size, opt, learning_rate, activation)
    modelEntity = dm.getModelEntity()
    modelEntity.setDesireEpochs(1)
    modelEntity.setDesireSteps(desire_steps)
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)

    step_time = 0
    step_count = 0
    remark = 3
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for i in range(num_batch):
            start_time = timer() 
            batch_offset = i * batch_size
            batch_end = (i+1) * batch_size
            #batch_list = imagenet_image_list[batch_offset:batch_end]
            X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
            #X_mini_batch_feed = load_imagenet_images(imagenet_t1k_img_path, batch_list, imgHeight, imgWidth)
            Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
            sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            end_time = timer()
            dur_time = end_time - start_time
            if (i+1) % remark == 0:
                #print('step %d / %d' %(i+1, num_batch))
                step_time += dur_time
                step_count += 1
            
    avg_step_time = step_time / step_count * 1000
    print(avg_step_time)


def pack_eval(conf_a, conf_b):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    
    model_type_a = conf_a[0]
    batch_size_a = conf_a[1]
    opt_a = conf_a[2]
    model_layer_a = conf_a[3]
    activation_a = conf_a[4]
    desire_steps_a = Y_data.shape[0] // batch_size_a

    model_type_b = conf_b[0]
    batch_size_b = conf_b[1]
    opt_b = conf_b[2]
    model_layer_b = conf_b[3]
    activation_b = conf_b[4]
    desire_steps_b = Y_data.shape[0] // batch_size_b
    
    maxBatchSize = max(batch_size_a, batch_size_b)

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=2)
    
    dm_a = DnnModel(model_type_a, str(net_instnace[0]), model_layer_a, imgWidth, imgHeight, numChannels, numClasses, batch_size_a, opt_a, learning_rate, activation_a)
    modelEntity_a = dm_a.getModelEntity()
    modelEntity_a.setDesireEpochs(1)
    modelEntity_a.setDesireSteps(desire_steps_a)
    modelLogit_a = modelEntity_a.build(features)
    trainOps_a = modelEntity_a.train(modelLogit_a, labels)

    dm_b = DnnModel(model_type_b, str(net_instnace[1]), model_layer_b, imgWidth, imgHeight, numChannels, numClasses, batch_size_b, opt_b, learning_rate, activation_b)
    modelEntity_b = dm_b.getModelEntity()
    modelEntity_b.setDesireEpochs(1)
    modelEntity_b.setDesireSteps(desire_steps_b)
    modelLogit_b = modelEntity_b.build(features)
    trainOps_b = modelEntity_b.train(modelLogit_b, labels)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    step_time = 0
    step_count = 0
    remark = 3
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // maxBatchSize
        for i in range(num_batch):
            #print('step %d / %d' %(i+1, num_batch))
            start_time = timer()
            batch_offset = i * maxBatchSize
            batch_end = (i+1) * maxBatchSize
            #batch_list = imagenet_image_list[batch_offset:batch_end]
            X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
            #X_mini_batch_feed = load_imagenet_images(imagenet_t1k_img_path, batch_list, imgHeight, imgWidth)
            Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
            sess.run([trainOps_a, trainOps_b], feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            end_time = timer()
            dur_time = end_time - start_time
            if (i+1) % remark == 0:
                step_time += dur_time
                step_count += 1

    avg_step_time = step_time / step_count * 1000
    print(avg_step_time)

if __name__ == "__main__":
    conf_list = gen_confs()
    #print(conf_list)
    #print(len(conf_list))
    parser = argparse.ArgumentParser()
    parser.add_argument("--packmode", action="store_true", default=False, help="use pack or single")
    parser.add_argument('--pack', help='indicate pack model id')
    parser.add_argument('--single', help='indicate pack model id')
    args = parser.parse_args()

    packmode = args.packmode
    if packmode == True:
        pack_list = args.pack
        conf_a_idx = pack_list.split(',')[0]
        conf_b_idx = pack_list.split(',')[1]
        conf_a = conf_list[int(conf_a_idx)]
        conf_b = conf_list[int(conf_b_idx)]
        pack_eval(conf_a, conf_b)

    else:
        single_model_idx = args.single
        conf_model = conf_list[int(single_model_idx)]
        single_eval(conf_model)

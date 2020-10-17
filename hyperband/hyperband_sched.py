import tensorflow as tf
import numpy as np
import itertools
from operator import itemgetter
from datetime import datetime
import sys
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe
import config as cfg_yml

from img_utils import * 
from dnn_model import DnnModel

imgWidth = cfg_yml.imgWidth
imgHeight = cfg_yml.imgHeight
numChannels = cfg_yml.numChannels
numClasses = cfg_yml.numClasses
rand_seed = cfg_yml.rand_seed

batch_size = cfg_yml.batch_size
opt_conf = cfg_yml.opt_conf
model_layer = cfg_yml.model_layer
activation = cfg_yml.activation
learning_rate = cfg_yml.learning_rate
model_type = cfg_yml.model_type

mnist_train_img_path = cfg_yml.mnist_train_img_path
mnist_train_label_path = cfg_yml.mnist_train_label_path
mnist_t10k_img_path = cfg_yml.mnist_t10k_img_path
mnist_t10k_label_path = cfg_yml.mnist_t10k_label_path

cifar_10_path = cfg_yml.cifar_10_path

imagenet_t10k_bin_path = cfg_yml.imagenet_t10k_bin_path
imagenet_t10k_label_path = cfg_yml.imagenet_t10k_label_path
imagenet_t1k_bin_path = cfg_yml.imagenet_t1k_bin_path
imagenet_t1k_label_path = cfg_yml.imagenet_t1k_label_path

def get_params(n_conf):
    all_conf = [model_type, batch_size, opt_conf, model_layer, learning_rate, activation]
    hp_conf = list(itertools.product(*all_conf))
    np.random.seed(rand_seed)
    idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf, replace=False)
    rand_conf = itemgetter(*idx_list)(hp_conf)

    return rand_conf

def run_params_pack_knn(confs, epochs, conn):
    seed = np.random.randint(rand_seed)
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    #X_data = load_mnist_image(mnist_train_img_path, seed)
    #Y_data = load_mnist_label_onehot(mnist_train_label_path, seed)
    #X_data_eval = load_mnist_image(mnist_t10k_img_path, seed)
    #Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path, seed)
    X_data, Y_data = load_cifar_train(cifar_10_path, seed)
    X_data_eval, Y_data_eval = load_cifar_test(cifar_10_path, seed)
    #X_data = load_imagenet_bin_pickle(imagenet_t10k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data = load_imagenet_labels_onehot(imagenet_t10k_label_path, numClasses)
    #X_data_eval = load_imagenet_bin_pickle(imagenet_t1k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data_eval = load_imagenet_labels_onehot(imagenet_t1k_label_path, numClasses)


    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))
    
    desire_epochs = epochs

    entity_pack = []
    train_pack = []
    eval_pack = [] 
    acc_pack = []
    batch_size_set = set()

    max_bs = np.NINF

    for cidx, cf in enumerate(confs):
        model_type = cf[0]
        batch_size = cf[1]
        batch_size_set.add(batch_size)
        opt = cf[2]
        model_layer = cf[3]
        learning_rate = cf[4]
        activation = cf[5]

        desire_steps = Y_data.shape[0] // batch_size
        dm = DnnModel(model_type, str(net_instnace[cidx]), model_layer, imgWidth, imgHeight, numChannels, numClasses, batch_size, opt, learning_rate, activation)
        modelEntity = dm.getModelEntity()
        modelEntity.setDesireEpochs(desire_epochs)
        modelEntity.setDesireSteps(desire_steps)
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        entity_pack.append(modelEntity)
        train_pack.append(trainOps)
        eval_pack.append(evalOps)

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)
        
        complete_flag = False

        while len(train_pack) != 0:
            num_steps = Y_data.shape[0] // max_bs
            for i in range(num_steps):
                print('step %d / %d' %(i+1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i+1) * max_bs
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(train_pack, feed_dict = {features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                for me in entity_pack:
                    me.setCurStep()
                    if me.isCompleteTrain():
                        print("model has been trained completely:",me.getModelInstance())
                        sess.run(me.setBatchSize(Y_data_eval.shape[0]))
                        train_pack.remove(me.getTrainOp())
                        complete_flag = True   
                
                if len(train_pack) == 0:
                    break
                
                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break
    
        #print("models have been training this run, start to evaluate")
        for ep in eval_pack:
            #num_steps = Y_data.shape[0] // max_bs
            acc_arg = ep.eval({features: X_data_eval, labels: Y_data_eval})
            #acc_arg = sess.run(ep, feed_dict = {features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            acc_pack.append(acc_arg)
            #print(acc_arg)
        
    conn.send(acc_pack)
    conn.close()
    print("Accuracy:", acc_pack)

def run_params_pack_random(confs, epochs, conn):
    seed = np.random.randint(rand_seed)
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    
    #X_data = load_mnist_image(mnist_train_img_path, seed)
    #Y_data = load_mnist_label_onehot(mnist_train_label_path, seed)
    #X_data_eval = load_mnist_image(mnist_t10k_img_path, seed)
    #Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path, seed)
    X_data, Y_data = load_cifar_train(cifar_10_path, seed)
    X_data_eval, Y_data_eval = load_cifar_test(cifar_10_path, seed)
    #X_data = load_imagenet_bin_pickle(imagenet_t10k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data = load_imagenet_labels_onehot(imagenet_t10k_label_path, numClasses)
    #X_data_eval = load_imagenet_bin_pickle(imagenet_t1k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data_eval = load_imagenet_labels_onehot(imagenet_t1k_label_path, numClasses)


    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))
    
    desire_epochs = epochs
    
    entity_pack = []
    train_pack = []
    eval_pack = [] 
    acc_pack = []
    batch_size_set = set()

    max_bs = np.NINF

    for cidx, cf in enumerate(confs):
        model_type = cf[0]
        batch_size = cf[1]
        batch_size_set.add(batch_size)
        opt = cf[2]
        model_layer = cf[3]
        learning_rate = cf[4]
        activation = cf[5]
        
        desire_steps = Y_data.shape[0] // batch_size
        dm = DnnModel(model_type, str(net_instnace[cidx]), model_layer, imgWidth, imgHeight, numChannels, numClasses, batch_size, opt, learning_rate, activation)
        modelEntity = dm.getModelEntity()
        modelEntity.setDesireEpochs(desire_epochs)
        modelEntity.setDesireSteps(desire_steps)
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        entity_pack.append(modelEntity)
        train_pack.append(trainOps)
        eval_pack.append(evalOps)

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)
        
        complete_flag = False

        while len(train_pack) != 0:
            num_steps = Y_data.shape[0] // max_bs
            for i in range(num_steps):
                print('step %d / %d' %(i+1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i+1) * max_bs
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(train_pack, feed_dict = {features: X_mini_batch_feed, labels: Y_mini_batch_feed})
                for me in entity_pack:
                    me.setCurStep()
                    if me.isCompleteTrain():
                        print("model has been trained completely:",me.getModelInstance())
                        sess.run(me.setBatchSize(Y_data_eval.shape[0]))
                        train_pack.remove(me.getTrainOp())
                        complete_flag = True   
                
                if len(train_pack) == 0:
                    break
                
                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break
    
        print("models have been training this run, start to evaluate")
        for ep in eval_pack:
            #num_steps = Y_data.shape[0] // max_bs
            acc_arg = ep.eval({features: X_data_eval, labels: Y_data_eval})
            #acc_arg = sess.run(ep, feed_dict = {features: X_mini_batch_feed, labels: Y_mini_batch_feed})
            acc_pack.append(acc_arg)
            #print(acc_arg)
        
    conn.send(acc_pack)
    conn.close()
    print("Accuracy:", acc_pack)
        
def run_params_pack_bs(batch_size, confs, epochs, conn):
    seed = np.random.randint(rand_seed)
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    #X_data = load_mnist_image(mnist_train_img_path, seed)
    #Y_data = load_mnist_label_onehot(mnist_train_label_path, seed)
    #X_data_eval = load_mnist_image(mnist_t10k_img_path, seed)
    #Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path, seed)
    X_data, Y_data = load_cifar_train(cifar_10_path, seed)
    X_data_eval, Y_data_eval = load_cifar_test(cifar_10_path, seed)
    #X_data = load_imagenet_bin_pickle(imagenet_t10k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data = load_imagenet_labels_onehot(imagenet_t10k_label_path, numClasses)
    #X_data_eval = load_imagenet_bin_pickle(imagenet_t1k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data_eval = load_imagenet_labels_onehot(imagenet_t1k_label_path, numClasses)

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))

    setbs_pack = []
    train_pack = []
    eval_pack = [] 
    acc_pack = []

    for cidx, civ in enumerate(confs):
        model_type = civ[0]
        opt = civ[2]
        model_layer = civ[3]
        learning_rate = civ[4]
        activation = civ[5]
                
        dm = DnnModel(model_type, str(net_instnace[cidx]), model_layer, imgWidth, imgHeight, numChannels, numClasses, batch_size, opt, learning_rate, activation)
        modelEntity = dm.getModelEntity()
        modelLogit = modelEntity.build(features)
        trainOps = modelEntity.train(modelLogit, labels)
        evalOps = modelEntity.evaluate(modelLogit, labels)
        setbs_pack.append(modelEntity.setBatchSize(Y_data_eval.shape[0]))
        train_pack.append(trainOps)
        eval_pack.append(evalOps)

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for e in range(epochs):
            for i in range(num_batch):
                #print('epoch %d / %d, step %d / %d' %(e+1, epochs, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(train_pack, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        
        sess.run(setbs_pack)
        for evalOps in eval_pack:
            acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
            acc_pack.append(acc_arg)
        
        conn.send(acc_pack)
        conn.close()
        print("Accuracy:", acc_pack)
    
def run_params(hyper_params, epochs, conn):
    seed = np.random.randint(rand_seed)
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])
    
    #X_data = load_mnist_image(mnist_train_img_path, seed)
    #Y_data = load_mnist_label_onehot(mnist_train_label_path, seed)
    #X_data_eval = load_mnist_image(mnist_t10k_img_path, seed)
    #Y_data_eval = load_mnist_label_onehot(mnist_t10k_label_path, seed)
    X_data, Y_data = load_cifar_train(cifar_10_path, seed)
    X_data_eval, Y_data_eval = load_cifar_test(cifar_10_path, seed)
    #X_data = load_imagenet_bin_pickle(imagenet_t10k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data = load_imagenet_labels_onehot(imagenet_t10k_label_path, numClasses)
    #X_data_eval = load_imagenet_bin_pickle(imagenet_t1k_bin_path, numChannels, imgWidth, imgHeight)
    #Y_data_eval = load_imagenet_labels_onehot(imagenet_t1k_label_path, numClasses)

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize)
    
    model_type = hyper_params[0]
    batch_size = hyper_params[1]
    opt = hyper_params[2]
    model_layer = hyper_params[3]
    learning_rate = hyper_params[4]
    activation = hyper_params[5]
    
    print("\n*** model: {} | batch size: {} | opt: {} | model layer: {} | learning rate: {} | activation: {} ***".format(model_type, batch_size, opt, model_layer, learning_rate, activation))

    dm = DnnModel(model_type, str(net_instnace), model_layer, imgWidth, imgHeight, numChannels, numClasses, batch_size, opt, learning_rate, activation)
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features)
    trainOps = modelEntity.train(modelLogit, labels)
    evalOps = modelEntity.evaluate(modelLogit, labels)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = Y_data.shape[0] // batch_size
        for e in range(epochs):
            for i in range(num_batch):
                #print('epoch %d / %d, step %d / %d' %(e+1, epochs, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i+1) * batch_size
                X_mini_batch_feed = X_data[batch_offset:batch_end,:,:,:]
                Y_mini_batch_feed = Y_data[batch_offset:batch_end,:]
                sess.run(trainOps, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        
        sess.run(modelEntity.setBatchSize(Y_data_eval.shape[0]))
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
        conn.send(acc_arg)
        conn.close()
        print("Accuracy:", acc_arg)


import tensorflow as tf
import numpy as np
import itertools
from operator import itemgetter
from datetime import datetime
import sys
import random as rd
from multiprocessing import Process, Pipe
from timeit import default_timer as timer
import config as cfg_yml
from img_utils import * 
from dnn_model import DnnModel

imgWidth = cfg_yml.imgWidth
imgHeight = cfg_yml.imgHeight
numChannels = cfg_yml.numChannels
numClasses = cfg_yml.numClasses
rand_seed = cfg_yml.rand_seed

batch_size_global = cfg_yml.batch_size
opt_conf_global = cfg_yml.opt_conf
model_layer_global = cfg_yml.model_layer
activation_global = cfg_yml.activation
learning_rate_global = cfg_yml.learning_rate
model_type_global = cfg_yml.model_type

mnist_train_img_path = cfg_yml.mnist_train_img_path
mnist_train_label_path = cfg_yml.mnist_train_label_path
mnist_t10k_img_path = cfg_yml.mnist_t10k_img_path
mnist_t10k_label_path = cfg_yml.mnist_t10k_label_path

cifar_10_path = cfg_yml.cifar_10_path

switcher={0:'model_type_global',1:'batch_size_global',2:'opt_conf_global',3:'model_layer_global',4:'learning_rate_global',5:'activation_global'}

# generate the configurations 
def gen_confs(n_conf):    
    all_conf = [model_type_global, batch_size_global, opt_conf_global, model_layer_global, learning_rate_global, activation_global]
    hp_conf = list(itertools.product(*all_conf))
    
    np.random.seed(rand_seed)
    idx_list = np.random.choice(np.arange(0, len(hp_conf)), n_conf, replace=False)
    rand_conf_list = list(itemgetter(*idx_list)(hp_conf))
    return rand_conf_list

def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 

def pack_trial_standalone(confs, t_dict, tr_dict):
    #print("confs:",confs)
    select_num = 3

    trial_dict = t_dict
    trial_result_dict = tr_dict
    confs_list = confs

    while len(confs_list) > 0:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        trial_packed_list.append(spoint)
        
        spoint_list = trial_result_dict.get(spoint) 
        ssl = sorted(spoint_list.items(), key=lambda kv: kv[1], reverse=True)
        
        if select_num <= len(ssl):
            for sidx in range(select_num):
                selected_conf = ssl[sidx][0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)

                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        else:
            for sidx in ssl:
                selected_conf = sidx[0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        confs_list.remove(spoint)
        trial_dict.pop(spoint)
        trial_result_dict.pop(spoint)
        for didx in trial_dict:
            trial_dict.get(didx).remove(spoint)
        for ridx in trial_result_dict:
            trial_result_dict.get(ridx).pop(spoint)

        print(trial_packed_list)

# evaluate conf_a and conf_b in seq and pack way 
def eval_conf_pair(conf_a, conf_b, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    seed = np.random.randint(100000, size=1)

    #X_data = load_mnist_image(mnist_train_img_path, seed)
    #Y_data = load_mnist_label_onehot(mnist_train_label_path, seed)
    cifar_train_data, cifar_train_label = load_cifar_train(cifar_10_path, seed)
    cifar_test_data, cifar_test_label = load_cifar_test(cifar_10_path, seed)

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=2)

    print(conf_a[0])
    print(conf_b)
    '''
    modelEntity_a = MLP("mlp_"+str(net_instnace[0]), conf_a[2], imgHeight, imgWidth, numChannels, conf_a[0], numClasses, conf_a[1])
    modelLogit_a = modelEntity_a.build(features)
    trainOps_a = modelEntity_a.train(modelLogit_a, labels)
    
    modelEntity_b = MLP("mlp_"+str(net_instnace[1]), conf_b[2], imgHeight, imgWidth, numChannels, conf_b[0], numClasses, conf_b[1])    
    modelLogit_b = modelEntity_b.build(features)
    trainOps_b = modelEntity_b.train(modelLogit_b, labels)

    train_pair = [trainOps_a,trainOps_b]

    pack_dur_time = -1
    seq_dur_time = -1

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        batch_size_a = conf_a[0]
        X_mini_batch_feed_a = X_data[0:batch_size_a,:,:,:]
        Y_mini_batch_feed_a = Y_data[0:batch_size_a,:]
        a_start_time = timer()
        sess.run(trainOps_a, feed_dict={features: X_mini_batch_feed_a, labels: Y_mini_batch_feed_a})
        a_end_time = timer()
        a_dur_time = a_end_time - a_start_time
        
        batch_size_b = conf_b[0]
        X_mini_batch_feed_b = X_data[0:batch_size_b,:,:,:]
        Y_mini_batch_feed_b = Y_data[0:batch_size_b,:]
        b_start_time = timer()
        sess.run(trainOps_b, feed_dict={features: X_mini_batch_feed_b, labels: Y_mini_batch_feed_b})
        b_end_time = timer()
        b_dur_time = b_end_time - b_start_time
        
        seq_dur_time = b_dur_time + a_dur_time

        batch_pack_size = max(conf_a[0],conf_b[0])
        X_mini_batch_feed = X_data[0:batch_pack_size,:,:,:]
        Y_mini_batch_feed = Y_data[0:batch_pack_size,:]
        pack_start_time = timer()
        sess.run(train_pair, feed_dict={features: X_mini_batch_feed, labels: Y_mini_batch_feed})
        pack_end_time = timer()
        pack_dur_time = pack_end_time - pack_start_time
        
    result = seq_dur_time - pack_dur_time
    '''
    result = np.random.randint()
    conn.send(result)
    conn.close()

# generate trails collection from list
def prep_trial(conf_list):
    trial_dict = dict()
    for cidx in conf_list:
        trial_dict[cidx] = list()
        for didx in conf_list:
            if cidx != didx:
                trial_dict[cidx].append(didx)
    return trial_dict

# sort the configurations according to trial result
def sort_conf_trial(trial_dict):
    trial_result_dict = dict()
    for key, value in trial_dict.items():
        trial_result_dict[key] = dict()
        for vidx in value:
            parent_conn, child_conn = Pipe()
            p = Process(target=eval_conf_pair, args=(key, vidx, child_conn))
            p.start()
            result = parent_conn.recv()
            parent_conn.close()
            p.join()

            trial_result_dict[key][vidx] = result
    
    return trial_result_dict

# sort the configurations according to batch size
def sort_conf_bs(trial_dict):
    trial_result_dict = dict()
    for key, value in trial_dict.items():
        trial_result_dict[key] = list()
        trial_distance_index = list()
        for vidx in value:
            distance = abs(key[0] - vidx[0])
            trial_distance_index.append(distance)
        sorted_value = sort_list(value, trial_distance_index)
        trial_result_dict[key] = sorted_value
    
    return trial_result_dict

# sort the configurations according to euclid distance
def sort_conf_euclid(trial_dict):
    trial_result_dict = dict()
    for key, value in trial_dict.items():
        trial_result_dict[key] = list()
        trial_distance_index = list()
        for vidx in value:
            distance = compute_euclid(key, vidx)
            trial_distance_index.append(distance)
        sorted_value = sort_list(value, trial_distance_index)
        trial_result_dict[key] = sorted_value
    return trial_result_dict

# compute the euclid distance of two configurations 
def compute_euclid(conf_a, conf_b):
    pair_euclid_distance = 0
    for cfi, cfv in enumerate(conf_a): 
        if isinstance(cfv, str):
            if conf_a[cfi] != conf_b[cfi]:
                pair_euclid_distance += 1
        else:
            conf_list = list(globals()[switcher.get(cfi)])
            conf_a_conf_idx = conf_list.index(cfv)
            conf_b_conf_idx = conf_list.index(conf_b[cfi])
            pair_euclid_distance += abs(conf_a_conf_idx - conf_b_conf_idx)  
    return pair_euclid_distance     
        
# schedule confs using trial-based knn
def knn_conf_trial(confs, topk):
    confs_list = list(confs)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = sort_conf_trial(trial_dict)
    
    trial_pack_collection = []

    while len(confs_list) > 0:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        #print("spoint:",spoint)
        trial_packed_list.append(spoint)
        
        spoint_list = trial_result_dict.get(spoint) 
        ssl = sorted(spoint_list.items(), key=lambda kv: kv[1], reverse=True)
        
        if topk <= len(ssl):
            for sidx in range(topk):
                selected_conf = ssl[sidx][0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)

                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        else:
            for sidx in ssl:
                selected_conf = sidx[0]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_dict.pop(selected_conf)
                trial_result_dict.pop(selected_conf)

                for didx in trial_dict:
                    trial_dict.get(didx).remove(selected_conf)
                for ridx in trial_result_dict:
                    trial_result_dict.get(ridx).pop(selected_conf)

        confs_list.remove(spoint)
        trial_dict.pop(spoint)
        trial_result_dict.pop(spoint)
        for didx in trial_dict:
            trial_dict.get(didx).remove(spoint)
        for ridx in trial_result_dict:
            trial_result_dict.get(ridx).pop(spoint)

        trial_pack_collection.append(trial_packed_list)

    return trial_pack_collection

# schedule confs using batchsize-based knn
def knn_conf_bs(confs, topk):
    confs_list = list(confs)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = sort_conf_bs(trial_dict)
    
    trial_pack_collection = []

    while len(confs_list) > 0:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        #print("spoint:",spoint)
        trial_packed_list.append(spoint)
        
        ssl = trial_result_dict.get(spoint) 
        #ssl = sorted(spoint_list.items(), key=lambda kv: kv[1], reverse=True)
        
        if topk <= len(ssl):
            for stidx in range(topk):
                selected_conf = ssl[stidx]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_result_dict.pop(selected_conf)
                
                for ridx in trial_result_dict:
                    if ridx != spoint:
                        trial_result_dict.get(ridx).remove(selected_conf)

            trial_result_dict[spoint] = trial_result_dict[spoint][topk:]

        else:
            for sidx in ssl:
                selected_conf = sidx
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_result_dict.pop(selected_conf)

                for ridx in trial_result_dict:
                    if ridx != spoint:
                        trial_result_dict.get(ridx).remove(selected_conf)

            trial_result_dict[spoint] = trial_result_dict[spoint][topk:]

        confs_list.remove(spoint)
        trial_result_dict.pop(spoint)
        for ridx in trial_result_dict:
            trial_result_dict.get(ridx).remove(spoint)

        trial_pack_collection.append(trial_packed_list)

    return trial_pack_collection
    
# schedule confs using euclid-based knn
def knn_conf_euclid(confs, topk):
    confs_list = list(confs)
    print(confs_list)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = sort_conf_euclid(trial_dict)
    
    trial_pack_collection = []
    
    while len(confs_list) > 0:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        trial_packed_list.append(spoint)
        ssl = trial_result_dict.get(spoint) 
        
        if topk <= len(ssl):
            for stidx in range(topk):
                selected_conf = ssl[stidx]
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_result_dict.pop(selected_conf)
                
                for ridx in trial_result_dict:
                    if ridx != spoint:
                        trial_result_dict.get(ridx).remove(selected_conf)

            trial_result_dict[spoint] = trial_result_dict[spoint][topk:]

        else:
            for sidx in ssl:
                selected_conf = sidx
                #print("selected_conf:",selected_conf)
                trial_packed_list.append(selected_conf)
                confs_list.remove(selected_conf)
                trial_result_dict.pop(selected_conf)

                for ridx in trial_result_dict:
                    if ridx != spoint:
                        trial_result_dict.get(ridx).remove(selected_conf)

            trial_result_dict[spoint] = trial_result_dict[spoint][topk:]

        confs_list.remove(spoint)
        trial_result_dict.pop(spoint)
        for ridx in trial_result_dict:
            trial_result_dict.get(ridx).remove(spoint)

        trial_pack_collection.append(trial_packed_list)

    return trial_pack_collection


# unify knn method, later 
def knn_conf(confs, topk, sort_method):
    confs_list = list(confs)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = sort_method(trial_dict)
    print(trial_result_dict)


if __name__ == "__main__":
    confs_num = 100
    confs_list = gen_confs(confs_num)
    print(confs_list)
    #knn_conf(confs_list,1, sort_conf_euclid)
    #knn_conf_euclid(confs_list, 3)
    #trial_dict = prep_trial(confs_list)
    #trial_result_dict = sort_conf_trial(trial_dict)
    #pack_trial_standalone(confs_list, trial_dict, trial_result_dict)


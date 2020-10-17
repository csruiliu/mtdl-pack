import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(".."))

import config.config_parameter as cfg_para
import config.config_path as cfg_path
from models.model_importer import ModelImporter
from utils.utils_img_func import load_imagenet_raw, load_imagenet_labels_onehot, load_cifar10_keras, load_mnist_image, load_mnist_label_onehot

hyperband_rand_seed = cfg_para.hyperband_random_seed
hyperband_train_dataset = cfg_para.hyperband_train_dataset

if hyperband_train_dataset == 'imagenet':
    train_img_path = cfg_path.imagenet_t50k_img_raw_path
    train_label_path = cfg_path.imagenet_t50k_label_path
    test_img_path = cfg_path.imagenet_t1k_img_raw_path
    test_label_path = cfg_path.imagenet_t1k_label_path

    img_width = cfg_para.img_width_imagenet
    img_height = cfg_para.img_height_imagenet
    num_channel = cfg_para.num_channels_rgb
    num_class = cfg_para.num_class_imagenet

    train_label = load_imagenet_labels_onehot(train_label_path, num_class)
    test_label = load_imagenet_labels_onehot(test_label_path, num_class)

elif hyperband_train_dataset == 'cifar10':
    img_width = cfg_para.img_width_cifar10
    img_height = cfg_para.img_height_cifar10
    num_channel = cfg_para.num_channels_rgb
    num_class = cfg_para.num_class_cifar10

    train_feature, train_label, test_feature, test_label = load_cifar10_keras()

elif hyperband_train_dataset == 'mnist':
    img_width = cfg_para.img_width_mnist
    img_height = cfg_para.img_height_mnist
    num_channel = cfg_para.num_channels_bw
    num_class = cfg_para.num_class_mnist

    train_img_path = cfg_path.mnist_train_img_path
    train_label_path = cfg_path.mnist_train_label_path
    test_img_path = cfg_path.mnist_test_10k_img_path
    test_label_path = cfg_path.mnist_test_10k_label_path

    train_feature = load_mnist_image(train_img_path)
    train_label = load_mnist_image(test_img_path)
    test_feature = load_mnist_image(test_img_path)
    test_label = load_mnist_label_onehot(test_label_path)


def hyperband_original(hyper_params, epochs):
    graph = tf.Graph()
    with graph.as_default():
        features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
        labels = tf.placeholder(tf.int64, [None, num_class])

        dt = datetime.now()
        np.random.seed(dt.microsecond)
        net_instnace = np.random.randint(sys.maxsize)

        model_arch = hyper_params[0]
        model_type = model_arch.split('-')[0]
        model_layer = int(model_arch.split('-')[1])
        batch_size = hyper_params[1]
        opt = hyper_params[2]
        learning_rate = hyper_params[3]
        activation = hyper_params[4]

        print("\n** model: {} | batch size: {} | opt: {} | model layer: {} | learn rate: {} | act: {} **".format(
            model_type, batch_size, opt, model_layer, learning_rate, activation))

        dm = ModelImporter(model_type, str(net_instnace), model_layer, img_height, img_width, num_channel, num_class,
                           batch_size, opt, learning_rate, activation, batch_padding=False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)

    if hyperband_train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // batch_size
        for e in range(epochs):
            for i in range(num_batch):
                # print('epoch %d / %d, step %d / %d' %(e+1, epochs, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size

                if hyperband_train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature[batch_offset:batch_end]

                train_label_batch = train_label[batch_offset:batch_end]

                sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

        if hyperband_train_dataset == 'imagenet':
            acc_sum = 0
            imagenet_batch_size_eval = 50
            num_batch_eval = test_label.shape[0] // imagenet_batch_size_eval
            test_image_list = sorted(os.listdir(test_img_path))
            for n in range(num_batch_eval):
                batch_offset = n * imagenet_batch_size_eval
                batch_end = (n + 1) * imagenet_batch_size_eval
                test_batch_list = test_image_list[batch_offset:batch_end]
                test_feature_batch = load_imagenet_raw(test_img_path, test_batch_list, img_height, img_width)
                test_label_batch = test_label[batch_offset:batch_end]
                acc_batch = sess.run(eval_op, feed_dict={features: test_feature_batch, labels: test_label_batch})
                acc_sum += acc_batch
            acc_avg = acc_sum / num_batch_eval
        else:
            acc_avg = sess.run(eval_op, feed_dict={features: test_feature, labels: test_label})

    print("Accuracy:", acc_avg)
    return acc_avg


def hyperband_pack_bs(batch_size, confs, epochs):
    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))

    train_pack = list()
    eval_pack = list()
    acc_pack = list()

    for cidx, civ in enumerate(confs):
        model_type = civ[0]
        opt = civ[2]
        model_layer = civ[3]
        learning_rate = civ[4]
        activation = civ[5]

        dm = ModelImporter(model_type, str(net_instnace), model_layer, img_height, img_width, num_channel, num_class,
                           batch_size, opt, learning_rate, activation, batch_padding=False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)
        train_pack.append(train_op)
        eval_pack.append(eval_op)

    if hyperband_train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label.shape[0] // batch_size
        for e in range(epochs):
            for i in range(num_batch):
                # print('epoch %d / %d, step %d / %d' %(e+1, epochs, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size

                if hyperband_train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature[batch_offset:batch_end]

                train_label_batch = train_label[batch_offset:batch_end]

                sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})

        if hyperband_train_dataset == 'imagenet':
            acc_sum = 0
            imagenet_batch_size_eval = 50
            num_batch_eval = test_label.shape[0] // imagenet_batch_size_eval
            test_image_list = sorted(os.listdir(test_img_path))
            for eval_op in eval_pack:
                for n in range(num_batch_eval):
                    batch_offset = n * imagenet_batch_size_eval
                    batch_end = (n + 1) * imagenet_batch_size_eval
                    test_batch_list = test_image_list[batch_offset:batch_end]
                    test_feature_batch = load_imagenet_raw(test_img_path, test_batch_list, img_height, img_width)
                    test_label_batch = test_label[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op, feed_dict={features: test_feature_batch, labels: test_label_batch})
                    acc_sum += acc_batch
                acc_avg = acc_sum / num_batch_eval
                acc_pack.append(acc_avg)
        else:
            for eval_op in eval_pack:
                acc_avg = sess.run(eval_op, feed_dict={features: test_feature, labels: test_label})
                acc_pack.append(acc_avg)

    print("Accuracy:", acc_pack)
    return acc_pack


def hyperband_pack_random(confs, epochs):
    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))

    desire_epochs = epochs

    entity_pack = list()
    train_pack = list()
    eval_pack = list()
    acc_pack = list()
    batch_size_set = set()

    for cidx, cf in enumerate(confs):
        model_type = cf[0]
        batch_size = cf[1]
        batch_size_set.add(batch_size)
        opt = cf[2]
        model_layer = cf[3]
        learning_rate = cf[4]
        activation = cf[5]

        desire_steps = train_label.shape[0] // batch_size
        dm = ModelImporter(model_type, str(net_instnace[cidx]), model_layer, img_height, img_width, num_channel,
                           num_class, batch_size, opt, learning_rate, activation, batch_padding=True)
        model_entity = dm.get_model_entity()

        model_entity.set_desire_epochs(desire_epochs)
        model_entity.set_desire_steps(desire_steps)
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)
        entity_pack.append(model_entity)
        train_pack.append(train_op)
        eval_pack.append(eval_op)

    if hyperband_train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)

        complete_flag = False

        while len(train_pack) != 0:
            num_steps = train_label.shape[0] // max_bs
            for i in range(num_steps):
                print('step %d / %d' % (i + 1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i + 1) * max_bs

                if hyperband_train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature[batch_offset:batch_end]
                train_label_batch = train_label[batch_offset:batch_end]

                sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})

                for me in entity_pack:
                    me.set_current_step()
                    if me.isCompleteTrain():
                        print("model has been trained completely:", me.get_model_instance_name())
                        sess.run(me.set_batch_size(test_label.shape[0]))
                        train_pack.remove(me.get_train_op())
                        complete_flag = True

                if len(train_pack) == 0:
                    break

                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break

        print("Start to evaluate")
        if hyperband_train_dataset == 'imagenet':
            acc_sum = 0
            imagenet_batch_size_eval = 50
            num_batch_eval = test_label.shape[0] // imagenet_batch_size_eval
            test_image_list = sorted(os.listdir(test_img_path))
            for eval_op in eval_pack:
                for n in range(num_batch_eval):
                    batch_offset = n * imagenet_batch_size_eval
                    batch_end = (n + 1) * imagenet_batch_size_eval
                    test_batch_list = test_image_list[batch_offset:batch_end]
                    test_feature_batch = load_imagenet_raw(test_img_path, test_batch_list, img_height, img_width)
                    test_label_batch = test_label[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op, feed_dict={features: test_feature_batch, labels: test_label_batch})
                    acc_sum += acc_batch
                acc_avg = acc_sum / num_batch_eval
                acc_pack.append(acc_avg)
        else:
            for eval_op in eval_pack:
                acc_avg = sess.run(eval_op, feed_dict={features: test_feature, labels: test_label})
                acc_pack.append(acc_avg)

    print("Accuracy:", acc_pack)
    return acc_pack


def hyperband_pack_knn(confs, epochs):
    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))
    
    desire_epochs = epochs

    entity_pack = list()
    train_pack = list()
    eval_pack = list()
    acc_pack = list()
    batch_size_set = set()

    for cidx, cf in enumerate(confs):
        model_type = cf[0]
        batch_size = cf[1]
        batch_size_set.add(batch_size)
        opt = cf[2]
        model_layer = cf[3]
        learning_rate = cf[4]
        activation = cf[5]

        desire_steps = train_label.shape[0] // batch_size

        dm = ModelImporter(model_type, str(net_instnace[cidx]), model_layer, img_height, img_width, num_channel,
                           num_class, batch_size, opt, learning_rate, activation, batch_padding=True)

        model_entity = dm.get_model_entity()
        model_entity.set_desire_epochs(desire_epochs)
        model_entity.set_desire_steps(desire_steps)
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)
        entity_pack.append(model_entity)
        train_pack.append(train_op)
        eval_pack.append(eval_op)

    if hyperband_train_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_img_path))

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)
        
        complete_flag = False

        while len(train_pack) != 0:
            num_steps = train_label.shape[0] // max_bs
            for i in range(num_steps):
                print('step %d / %d' %(i+1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i+1) * max_bs

                if hyperband_train_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_img_path, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature[batch_offset:batch_end]
                train_label_batch = train_label[batch_offset:batch_end]

                sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})
                for me in entity_pack:
                    me.setCurStep()
                    if me.isCompleteTrain():
                        print("model has been trained completely:{}".format(me.get_model_instance()))
                        sess.run(me.setBatchSize(test_label.shape[0]))
                        train_pack.remove(me.get_train_op())
                        complete_flag = True   
                
                if len(train_pack) == 0:
                    break
                
                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break

        print("Start to evaluate")
        if hyperband_train_dataset == 'imagenet':
            acc_sum = 0
            imagenet_batch_size_eval = 50
            num_batch_eval = test_label.shape[0] // imagenet_batch_size_eval
            test_image_list = sorted(os.listdir(test_img_path))
            for eval_op in eval_pack:
                for n in range(num_batch_eval):
                    batch_offset = n * imagenet_batch_size_eval
                    batch_end = (n + 1) * imagenet_batch_size_eval
                    test_batch_list = test_image_list[batch_offset:batch_end]
                    test_feature_batch = load_imagenet_raw(test_img_path, test_batch_list, img_height, img_width)
                    test_label_batch = test_label[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op, feed_dict={features: test_feature_batch, labels: test_label_batch})
                    acc_sum += acc_batch
                acc_avg = acc_sum / num_batch_eval
                acc_pack.append(acc_avg)
        else:
            for eval_op in eval_pack:
                acc_avg = sess.run(eval_op, feed_dict={features: test_feature, labels: test_label})
                acc_pack.append(acc_avg)

    print("Accuracy:", acc_pack)
    return acc_pack

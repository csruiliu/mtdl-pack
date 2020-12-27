import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import sys

import pack.config.config_parameter as cfg_para
from pack.core.dataset_loader import load_dataset_para, load_train_dataset, load_eval_dataset
from pack.core.model_importer import ModelImporter
from pack.tools.img_tool import load_imagenet_raw


def evaluate_pack_model(tf_sess, feature_ph, label_ph, pack_model):
    print("start to evaluate")
    hyperband_dataset = cfg_para.hyperband_train_dataset
    img_width, img_height, _, _ = load_dataset_para(hyperband_dataset)
    feature_input, label_input = load_eval_dataset(hyperband_dataset)

    acc_pack = list()

    if hyperband_dataset == 'imagenet':
        acc_sum = 0
        imagenet_batch_size_eval = 50
        num_batch_eval = label_input.shape[0] // imagenet_batch_size_eval
        test_image_list = sorted(os.listdir(feature_input))
        for eval_op in pack_model:
            for n in range(num_batch_eval):
                batch_offset = n * imagenet_batch_size_eval
                batch_end = (n + 1) * imagenet_batch_size_eval
                eval_batch_list = test_image_list[batch_offset:batch_end]
                eval_feature_batch = load_imagenet_raw(feature_input, eval_batch_list, img_height, img_width)
                eval_label_batch = label_input[batch_offset:batch_end]
                acc_batch = tf_sess.run(eval_op, feed_dict={feature_ph: eval_feature_batch,
                                                            label_ph: eval_label_batch})
                acc_sum += acc_batch
            acc_avg = acc_sum / num_batch_eval
            acc_pack.append(acc_avg)
    else:
        for eval_op in pack_model:
            acc_avg = tf_sess.run(eval_op, feed_dict={feature_ph: feature_input, label_ph: label_input})
            acc_pack.append(acc_avg)

    return acc_pack


def hyperband_original(hyper_params, epochs):
    # load dataset
    hyperband_dataset = cfg_para.hyperband_train_dataset

    img_width, img_height, num_channel, num_class = load_dataset_para(hyperband_dataset)
    train_feature_input, train_label_input = load_train_dataset(hyperband_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(hyperband_dataset)

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

        print("\n** model: {} | batch size: {} | opt: {} | model layer: {} | learn rate: {} | act: {} **"
              .format(model_type, batch_size, opt, model_layer, learning_rate, activation))

        dm = ModelImporter(model_type, str(net_instnace), model_layer, img_height,
                           img_width, num_channel, num_class, batch_size, opt,
                           learning_rate, activation, batch_padding=False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)

    if hyperband_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label_input.shape[0] // batch_size
        for e in range(epochs):
            for i in range(num_batch):
                # print('epoch %d / %d, step %d / %d' %(e+1, epochs, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size

                if hyperband_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(hyperband_dataset, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature_input[batch_offset:batch_end]

                train_label_batch = train_label_input[batch_offset:batch_end]

                sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

        if hyperband_dataset == 'imagenet':
            acc_sum = 0
            imagenet_batch_size_eval = 50
            num_batch_eval = eval_label_input.shape[0] // imagenet_batch_size_eval
            test_image_list = sorted(os.listdir(eval_feature_input))
            for n in range(num_batch_eval):
                batch_offset = n * imagenet_batch_size_eval
                batch_end = (n + 1) * imagenet_batch_size_eval
                test_batch_list = test_image_list[batch_offset:batch_end]
                test_feature_batch = load_imagenet_raw(eval_feature_input, test_batch_list, img_height, img_width)
                test_label_batch = eval_label_input[batch_offset:batch_end]
                acc_batch = sess.run(eval_op, feed_dict={features: test_feature_batch, labels: test_label_batch})
                acc_sum += acc_batch
            acc_avg = acc_sum / num_batch_eval
        else:
            acc_avg = sess.run(eval_op, feed_dict={features: eval_feature_input, labels: eval_label_input})

    print("Accuracy:", acc_avg)
    return acc_avg


def hyperband_pack_bs(batch_size, confs, epochs):
    # load dataset
    hyperband_dataset = cfg_para.hyperband_train_dataset

    img_width, img_height, num_channel, num_class = load_dataset_para(hyperband_dataset)
    train_feature_input, train_label_input = load_train_dataset(hyperband_dataset)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))

    train_pack = list()
    eval_pack = list()

    for cidx, civ in enumerate(confs):
        model_arch = civ[0]
        model_type = model_arch.split('-')[0]
        model_layer = int(model_arch.split('-')[1])
        opt = civ[2]
        learning_rate = civ[3]
        activation = civ[4]

        dm = ModelImporter(model_type, str(net_instnace[cidx]), model_layer,
                           img_height, img_width, num_channel, num_class,
                           batch_size, opt, learning_rate, activation, batch_padding=False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)
        train_pack.append(train_op)
        eval_pack.append(eval_op)

    if hyperband_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = train_label_input.shape[0] // batch_size
        for e in range(epochs):
            for i in range(num_batch):
                # print('epoch %d / %d, step %d / %d' %(e+1, epochs, i+1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size

                if hyperband_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_feature_input, batch_list,
                                                            img_height, img_width)
                else:
                    train_feature_batch = train_feature_input[batch_offset:batch_end]

                train_label_batch = train_label_input[batch_offset:batch_end]

                sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})

        acc_pack = evaluate_pack_model(sess, features, labels, eval_pack)

    print("Accuracy:", acc_pack)
    return acc_pack


def hyperband_pack_random(confs, epochs):
    # load dataset
    hyperband_dataset = cfg_para.hyperband_train_dataset

    img_width, img_height, num_channel, num_class = load_dataset_para(hyperband_dataset)
    train_feature_input, train_label_input = load_train_dataset(hyperband_dataset)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dt = datetime.now()
    np.random.seed(dt.microsecond)
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))

    desire_epochs = epochs

    entity_pack = list()
    train_pack = list()
    eval_pack = list()
    batch_size_set = set()

    for cidx, cf in enumerate(confs):
        model_arch = cf[0]
        model_type = model_arch.split('-')[0]
        model_layer = int(model_arch.split('-')[1])
        batch_size = cf[1]
        batch_size_set.add(batch_size)
        opt = cf[2]
        learning_rate = cf[3]
        activation = cf[4]

        desire_steps = train_label_input.shape[0] // batch_size
        dm = ModelImporter(model_type, str(net_instnace[cidx]), model_layer,
                           img_height, img_width, num_channel, num_class,
                           batch_size, opt, learning_rate, activation, batch_padding=True)
        model_entity = dm.get_model_entity()

        model_entity.set_desire_epochs(desire_epochs)
        model_entity.set_desire_steps(desire_steps)
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)
        entity_pack.append(model_entity)
        train_pack.append(train_op)
        eval_pack.append(eval_op)

    if hyperband_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)

        complete_flag = False

        while len(train_pack) != 0:
            num_steps = train_label_input.shape[0] // max_bs
            for i in range(num_steps):
                print('step %d / %d' % (i+1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i+1) * max_bs

                if hyperband_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_feature_input, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature_input[batch_offset:batch_end]
                train_label_batch = train_label_input[batch_offset:batch_end]

                sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})
                for me in entity_pack:
                    me.set_current_step()
                    if me.is_complete_train():
                        print("model has been trained completely:{}".format(me.get_model_instance_name()))
                        sess.run(me.set_batch_size(train_label_input.shape[0]))
                        train_pack.remove(me.get_train_op())
                        complete_flag = True

                if len(train_pack) == 0:
                    break

                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break

        acc_pack = evaluate_pack_model(sess, features, labels, eval_pack)

    print("Accuracy:", acc_pack)
    return acc_pack


def hyperband_pack_knn(confs, epochs):
    # load dataset
    hyperband_dataset = cfg_para.hyperband_train_dataset

    img_width, img_height, num_channel, num_class = load_dataset_para(hyperband_dataset)
    train_feature_input, train_label_input = load_train_dataset(hyperband_dataset)

    features = tf.placeholder(tf.float32, [None, img_width, img_height, num_channel])
    labels = tf.placeholder(tf.int64, [None, num_class])

    dt = datetime.now()
    np.random.seed(dt.microsecond)    
    net_instnace = np.random.randint(sys.maxsize, size=len(confs))
    
    desire_epochs = epochs

    entity_pack = list()
    train_pack = list()
    eval_pack = list()
    batch_size_set = set()

    for cidx, cf in enumerate(confs):
        model_arch = cf[0]
        model_type = model_arch.split('-')[0]
        model_layer = int(model_arch.split('-')[1])
        batch_size = cf[1]
        batch_size_set.add(batch_size)
        opt = cf[2]
        learning_rate = cf[3]
        activation = cf[4]

        desire_steps = train_label_input.shape[0] // batch_size

        dm = ModelImporter(model_type, str(net_instnace[cidx]), model_layer,
                           img_height, img_width, num_channel, num_class,
                           batch_size, opt, learning_rate, activation, batch_padding=True)

        model_entity = dm.get_model_entity()
        model_entity.set_desire_epochs(desire_epochs)
        model_entity.set_desire_steps(desire_steps)
        model_logit = model_entity.build(features, is_training=True)
        train_op = model_entity.train(model_logit, labels)
        eval_op = model_entity.evaluate(model_logit, labels)
        entity_pack.append(model_entity)
        train_pack.append(train_op)
        eval_pack.append(eval_op)

    if hyperband_dataset == 'imagenet':
        image_list = sorted(os.listdir(train_feature_input))

    config = tf.ConfigProto()
    config.allow_soft_placement = True   
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        max_bs = max(batch_size_set)
        
        complete_flag = False

        while len(train_pack) != 0:
            num_steps = train_label_input.shape[0] // max_bs
            for i in range(num_steps):
                print('step %d / %d' % (i+1, num_steps))
                batch_offset = i * max_bs
                batch_end = (i+1) * max_bs

                if hyperband_dataset == 'imagenet':
                    batch_list = image_list[batch_offset:batch_end]
                    train_feature_batch = load_imagenet_raw(train_feature_input, batch_list, img_height, img_width)
                else:
                    train_feature_batch = train_feature_input[batch_offset:batch_end]
                train_label_batch = train_label_input[batch_offset:batch_end]

                sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})
                for me in entity_pack:
                    me.set_current_step()
                    if me.is_complete_train():
                        print("model has been trained completely:{}".format(me.get_model_instance_name()))
                        sess.run(me.set_batch_size(train_label_input.shape[0]))
                        train_pack.remove(me.get_train_op())
                        complete_flag = True
                
                if len(train_pack) == 0:
                    break
                
                if complete_flag:
                    batch_size_set.discard(max_bs)
                    max_bs = max(batch_size_set)
                    complete_flag = False
                    break

        acc_pack = evaluate_pack_model(sess, features, labels, eval_pack)

    print("Accuracy:", acc_pack)
    return acc_pack

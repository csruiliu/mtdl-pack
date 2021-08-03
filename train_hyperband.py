import tensorflow as tf
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import itertools
import os
import sys
from operator import itemgetter
from math import log, ceil, floor

import config.config_parameter as cfg_para
from tools.knn_engine import KNNEngine
from tools.model_tool import ModelImporter
from tools.dataset_tool import load_dataset_para, load_train_dataset, load_eval_dataset, load_imagenet_raw


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


class Hyperband:
    def __init__(self, resource_conf, down_rate):
        # maximun budget for single configuration, i.e., maximum iterations per configuration in example
        self.R = resource_conf
        # defines configuration downsampling rate (default = 3)
        self.eta = down_rate
        # control how many runs
        self.s_max = floor(log(self.R, self.eta))
        # maximun budget for all configurations
        self.B = (self.s_max + 1) * self.R
        # list of results
        self.results = []

        # parameters for results
        self.counter = 0
        self.best_acc = np.NINF
        self.best_counter = -1

        # parameters for workload
        self.hp_model_arch = cfg_para.hyperband_model_type_list
        self.hp_batch_size = cfg_para.hyperband_batch_size_list
        self.hp_opt = cfg_para.hyperband_optimizer_list
        self.hp_learn_rate = cfg_para.hyperband_learn_rate_list
        self.hp_activation = cfg_para.hyperband_activation_list
        self.hp_random_seed = cfg_para.hyperband_random_seed

        # training dataset
        self.hp_dataset = cfg_para.hyperband_train_dataset
        (self.img_width,
         self.img_height,
         self.num_channel,
         self.num_class) = load_dataset_para(self.hp_dataset)

    def generate_hyperband_workload(self, sample_size):
        all_conf = [self.hp_model_arch, self.hp_batch_size, self.hp_opt, self.hp_learn_rate, self.hp_activation]
        hyperband_workload = list(itertools.product(*all_conf))
        np.random.seed(self.hp_random_seed)
        idx_list = np.random.choice(np.arange(0, len(hyperband_workload)), sample_size, replace=False)
        sampled_workload = itemgetter(*idx_list)(hyperband_workload)

        return sampled_workload

    def hyperband_original(self, hyper_params, epochs):
        train_feature_input, train_label_input = load_train_dataset(self.hp_dataset)
        eval_feature_input, eval_label_input = load_eval_dataset(self.hp_dataset)

        graph = tf.Graph()
        with graph.as_default():
            features = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.num_channel])
            labels = tf.placeholder(tf.int64, [None, self.num_class])

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

            dm = ModelImporter(model_type,
                               str(net_instnace),
                               model_layer,
                               self.img_height,
                               self.img_width,
                               self.num_channel,
                               self.num_class,
                               batch_size,
                               opt,
                               learning_rate,
                               activation,
                               batch_padding=False)
            model_entity = dm.get_model_entity()
            model_logit = model_entity.build(features, is_training=True)
            train_op = model_entity.train(model_logit, labels)
            eval_op = model_entity.evaluate(model_logit, labels)

        if self.hp_dataset == 'imagenet':
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

                    if self.hp_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(self.hp_dataset,
                                                                batch_list,
                                                                self.img_height,
                                                                self.img_width)
                    else:
                        train_feature_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]

                    sess.run(train_op, feed_dict={features: train_feature_batch, labels: train_label_batch})

            if self.hp_dataset == 'imagenet':
                acc_sum = 0
                imagenet_batch_size_eval = 50
                num_batch_eval = eval_label_input.shape[0] // imagenet_batch_size_eval
                test_image_list = sorted(os.listdir(eval_feature_input))
                for n in range(num_batch_eval):
                    batch_offset = n * imagenet_batch_size_eval
                    batch_end = (n + 1) * imagenet_batch_size_eval
                    test_batch_list = test_image_list[batch_offset:batch_end]
                    test_feature_batch = load_imagenet_raw(eval_feature_input,
                                                           test_batch_list,
                                                           self.img_height,
                                                           self.img_width)
                    test_label_batch = eval_label_input[batch_offset:batch_end]
                    acc_batch = sess.run(eval_op, feed_dict={features: test_feature_batch, labels: test_label_batch})
                    acc_sum += acc_batch
                acc_avg = acc_sum / num_batch_eval
            else:
                acc_avg = sess.run(eval_op, feed_dict={features: eval_feature_input, labels: eval_label_input})

        print(f'Accuracy: {acc_avg}')
        return acc_avg

    def hyperband_pack_bs(self, batch_size, confs, epochs):
        train_feature_input, train_label_input = load_train_dataset(self.hp_dataset)

        features = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.num_channel])
        labels = tf.placeholder(tf.int64, [None, self.num_class])

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

            dm = ModelImporter(model_type,
                               str(net_instnace[cidx]),
                               model_layer,
                               self.img_height,
                               self.img_width,
                               self.num_channel,
                               self.num_class,
                               batch_size,
                               opt,
                               learning_rate,
                               activation,
                               batch_padding=False)

            model_entity = dm.get_model_entity()
            model_logit = model_entity.build(features, is_training=True)
            train_op = model_entity.train(model_logit, labels)
            eval_op = model_entity.evaluate(model_logit, labels)
            train_pack.append(train_op)
            eval_pack.append(eval_op)

        if self.hp_dataset == 'imagenet':
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

                    if self.hp_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input,
                                                                batch_list,
                                                                self.img_height,
                                                                self.img_width)
                    else:
                        train_feature_batch = train_feature_input[batch_offset:batch_end]

                    train_label_batch = train_label_input[batch_offset:batch_end]

                    sess.run(train_pack, feed_dict={features: train_feature_batch, labels: train_label_batch})

            acc_pack = evaluate_pack_model(sess, features, labels, eval_pack)

        print(f'Accuracy: {acc_pack}')
        return acc_pack

    def hyperband_pack_random(self, confs, epochs):
        train_feature_input, train_label_input = load_train_dataset(self.hp_dataset)

        features = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.num_channel])
        labels = tf.placeholder(tf.int64, [None, self.num_class])

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
            dm = ModelImporter(model_type,
                               str(net_instnace[cidx]),
                               model_layer,
                               self.img_height,
                               self.img_width,
                               self.num_channel,
                               self.num_class,
                               batch_size,
                               opt,
                               learning_rate,
                               activation,
                               batch_padding=True)
            model_entity = dm.get_model_entity()

            model_entity.set_desire_epochs(desire_epochs)
            model_entity.set_desire_steps(desire_steps)
            model_logit = model_entity.build(features, is_training=True)
            train_op = model_entity.train(model_logit, labels)
            eval_op = model_entity.evaluate(model_logit, labels)
            entity_pack.append(model_entity)
            train_pack.append(train_op)
            eval_pack.append(eval_op)

        if self.hp_dataset == 'imagenet':
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
                    print('step %d / %d' % (i + 1, num_steps))
                    batch_offset = i * max_bs
                    batch_end = (i + 1) * max_bs

                    if self.hp_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input,
                                                                batch_list,
                                                                self.img_height,
                                                                self.img_width)
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

    def hyperband_pack_knn(self, confs, epochs):
        train_feature_input, train_label_input = load_train_dataset(self.hp_dataset)

        features = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.num_channel])
        labels = tf.placeholder(tf.int64, [None, self.num_class])

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

            dm = ModelImporter(model_type,
                               str(net_instnace[cidx]),
                               model_layer,
                               self.img_height,
                               self.img_width,
                               self.num_channel,
                               self.num_class,
                               batch_size, opt,
                               learning_rate,
                               activation,
                               batch_padding=True)

            model_entity = dm.get_model_entity()
            model_entity.set_desire_epochs(desire_epochs)
            model_entity.set_desire_steps(desire_steps)
            model_logit = model_entity.build(features, is_training=True)
            train_op = model_entity.train(model_logit, labels)
            eval_op = model_entity.evaluate(model_logit, labels)
            entity_pack.append(model_entity)
            train_pack.append(train_op)
            eval_pack.append(eval_op)

        if self.hp_dataset == 'imagenet':
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
                    print('step %d / %d' % (i + 1, num_steps))
                    batch_offset = i * max_bs
                    batch_end = (i + 1) * max_bs

                    if self.hp_dataset == 'imagenet':
                        batch_list = image_list[batch_offset:batch_end]
                        train_feature_batch = load_imagenet_raw(train_feature_input,
                                                                batch_list,
                                                                self.img_height,
                                                                self.img_width)
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

        print(f'Accuracy: {acc_pack}')
        return acc_pack

    def run_simulation(self):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.generate_hyperband_workload(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                print("==============================================================")
                # record all accuracy of current run for sorting
                val_acc = []
                for t in T:
                    result = {'acc': -1, 'counter': -1}
                    self.counter += 1
                    # generate random accuracy
                    acc = np.random.random()
                    val_acc.append(acc)

                    result['acc'] = acc
                    result['counter'] = self.counter
                    result['params'] = t

                    # record the best result
                    if self.best_acc < acc:
                        self.best_acc = acc
                        self.best_counter = self.counter
                    print("current run {}, acc: {:.5f} | best acc: {:.5f} (run {})\n".format(self.counter, acc,
                                                                                             self.best_acc,
                                                                                             self.best_counter))
                    self.results.append(result)

                # sort the result
                indices = np.argsort(val_acc)

                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

        return self.results

    def run_original(self):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * self.eta ** (-s)
            T = self.generate_hyperband_workload(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                list_acc = []

                for t in T:
                    result = {'acc': -1, 'counter': -1}
                    self.counter += 1
                    acc = self.hyperband_original(t, r_i)
                    result['acc'] = acc

                    list_acc.append(acc)
                    if self.best_acc < acc:
                        self.best_acc = acc
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['params'] = t

                    print("current run {}, acc: {:.5f} | best acc so far: {:.5f} (run {})\n"
                          .format(self.counter, acc, self.best_acc, self.best_counter))
                    self.results.append(result)

                indices = np.argsort(list_acc)
                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

        return self.results

    def run_pack_bs(self):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.generate_hyperband_workload(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                val_acc = []
                params_dict = dict()

                for t in T:
                    if t[1] in params_dict:
                        params_dict[t[1]].append(t)
                    else:
                        params_dict[t[1]] = []
                        params_dict[t[1]].append(t)

                for bs, conf in params_dict.items():
                    acc_pack = self.hyperband_pack_bs(bs, conf, r_i)
                    for aidx, acc in enumerate(acc_pack):
                        result = {'acc': -1}
                        val_acc.append(acc)
                        result['acc'] = acc
                        result['params'] = conf[aidx]

                        if self.best_acc < acc:
                            self.best_acc = acc
                            print("best accuracy so far: {:.5f} \n".format(self.best_acc))
                        self.results.append(result)

                indices = np.argsort(val_acc)
                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

        return self.results

    def run_pack_random(self, random_size):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.generate_hyperband_workload(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** i)
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))
                val_acc = []
                params_list = []
                num_para_list = ceil(len(T) / random_size)
                if num_para_list == 1:
                    params_list.append(T)
                else:
                    for npl in range(num_para_list - 1):
                        params_list.append(T[npl * random_size:(npl + 1) * random_size])
                    params_list.append(T[(num_para_list - 1) * random_size:])

                for pidx in range(num_para_list):
                    selected_confs = params_list[pidx]
                    acc_pack = self.hyperband_pack_random(selected_confs, r_i)
                    for idx, acc in enumerate(acc_pack):
                        result = {'acc': -1}
                        val_acc.append(acc)
                        result['acc'] = acc
                        result['params'] = []
                        for params in selected_confs[idx]:
                            result['params'].append(params)

                        if self.best_acc < acc:
                            self.best_acc = acc
                            print("best accuracy so far: {:.5f} \n".format(self.best_acc))
                        self.results.append(result)

                indices = np.argsort(val_acc)
                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

        return self.results

    def run_pack_knn(self, topk, knn_instance):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.generate_hyperband_workload(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print(f'\n*** {s} bracket | {n_i} configurations x {r_i} iterations each ***')

                val_acc = []
                trial_pack_collection = knn_instance.knn_conf_euclid(T, topk)

                for tpidx in trial_pack_collection:
                    acc_pack = self.hyperband_pack_knn(tpidx, r_i)

                    for idx, acc in enumerate(acc_pack):
                        result = {'acc': -1}
                        val_acc.append(acc)
                        result['acc'] = acc
                        result['params'] = []
                        for param in tpidx[idx]:
                            result['params'].append(param)

                        if self.best_acc < acc:
                            self.best_acc = acc
                            print("best accuracy so far: {:.5f} \n".format(self.best_acc))
                        self.results.append(result)

                indices = np.argsort(val_acc)
                T = [T[i] for i in indices]
                T = T[0:floor(n_i / self.eta)]

        return self.results


if __name__ == '__main__':
    # read configurations
    resource_conf = cfg_para.hyperband_resource_conf
    down_rate = cfg_para.hyperband_down_rate
    sch_policy = cfg_para.hyperband_schedule_policy
    hyperband_pack_rate = cfg_para.hyperband_pack_rate

    results = None

    start_time = timer()

    if sch_policy == 'none':
        hb = Hyperband(resource_conf, down_rate)
        results = hb.run_original()
    elif sch_policy == 'pack-bs':
        hb = Hyperband(resource_conf, down_rate)
        results = hb.run_pack_bs()
    elif sch_policy == 'pack-random':
        hb = Hyperband(resource_conf, down_rate)
        results = hb.run_pack_random(hyperband_pack_rate)
    elif sch_policy == 'pack-knn':
        hb = Hyperband(resource_conf, down_rate)
        knn_engine = KNNEngine(cfg_para.hyperband_model_type_list,
                               cfg_para.hyperband_batch_size_list,
                               cfg_para.hyperband_optimizer_list,
                               cfg_para.hyperband_learn_rate_list,
                               cfg_para.hyperband_activation_list)

        results = hb.run_pack_knn(hyperband_pack_rate, knn_engine)

    end_time = timer()

    dur_time = end_time - start_time
    print("{} total, best:\n".format(len(results)))
    best_hp = sorted(results, key=lambda x: x['acc'])[-1]
    print(best_hp)
    print('total exp time: {}'.format(dur_time))

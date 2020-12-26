from timeit import default_timer as timer
import numpy as np
import itertools
from operator import itemgetter
from math import log, ceil, floor
import os
import sys
sys.path.append(os.path.abspath(".."))

import config.config_parameter as cfg_para
from hyperband_sched_engine import hyperband_original, hyperband_pack_bs, hyperband_pack_random, hyperband_pack_knn
from knn_engine import knn_conf_bs, knn_conf_euclid


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
        
        self.counter = 0
        self.best_acc = np.NINF
        self.best_counter = -1

        self.hp_model_arch = cfg_para.hyperband_model_type_list
        self.hp_batch_size = cfg_para.hyperband_batch_size_list
        self.hp_opt = cfg_para.hyperband_optimizer_list
        self.hp_learn_rate = cfg_para.hyperband_learn_rate_list
        self.hp_activation = cfg_para.hyperband_activation_list
        self.hp_random_seed = cfg_para.hyperband_random_seed

    def generate_hyperband_workload(self, sample_size):
        all_conf = [self.hp_model_arch, self.hp_batch_size, self.hp_opt, self.hp_learn_rate, self.hp_activation]
        hyperband_workload = list(itertools.product(*all_conf))
        np.random.seed(self.hp_random_seed)
        idx_list = np.random.choice(np.arange(0, len(hyperband_workload)), sample_size, replace=False)
        sampled_workload = itemgetter(*idx_list)(hyperband_workload)

        return sampled_workload

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
                    acc = hyperband_original(t, r_i)
                    result['acc'] = acc

                    list_acc.append(acc)
                    if self.best_acc < acc:
                        self.best_acc = acc
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['params'] = t

                    print("current run {}, acc: {:.5f} | best acc so far: {:.5f} (run {})\n".format(self.counter, acc,
                                                                                                    self.best_acc,
                                                                                                    self.best_counter))
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
                    acc_pack = hyperband_pack_bs(bs, conf, r_i)
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
                    acc_pack = hyperband_pack_random(selected_confs, r_i)
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

    def run_pack_knn(self, topk, knn_method):
        for s in reversed(range(self.s_max + 1)):
            n = ceil(self.B / self.R / (s + 1) * (self.eta ** s))
            r = self.R * (self.eta ** (-s))
            T = self.generate_hyperband_workload(n)

            for i in range(s + 1):
                n_i = floor(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))
                print("\n*** {} bracket | {} configurations x {} iterations each ***".format(s, n_i, r_i))

                val_acc = []
                trial_pack_collection = knn_method(T, topk)

                for tpidx in trial_pack_collection:
                    acc_pack = hyperband_pack_knn(tpidx, r_i)

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


if __name__ == "__main__":
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
        results = hb.run_pack_knn(hyperband_pack_rate, knn_conf_euclid)

    end_time = timer()

    dur_time = end_time - start_time
    print("{} total, best:\n".format(len(results)))
    best_hp = sorted(results, key = lambda x: x['acc'])[-1]
    print(best_hp)
    print('total exp time: {}'.format(dur_time))

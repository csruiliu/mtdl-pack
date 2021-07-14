import random as rd


class KNNEngine:
    def __init__(self,
                 model_list,
                 batch_size_list,
                 opt_list,
                 learn_rate_list,
                 activation_list):

        self.model_list = model_list
        self.model_layer_list = list()

        for midx in model_list:
            self.model_layer_list.append(int(midx.split('-')[1]))

        self.batch_size_list = batch_size_list
        self.opt_list = opt_list
        self.learn_rate_list = learn_rate_list
        self.activation_list = activation_list

    @staticmethod
    def sort_list(list1, list2):
        zipped_pairs = zip(list2, list1)
        z = [x for _, x in sorted(zipped_pairs)]
        return z

    @staticmethod
    def sort_conf_bs(self, trial_dict):
        # sort the configurations according to batch size
        trial_result_dict = dict()
        for key, value in trial_dict.items():
            trial_result_dict[key] = list()
            trial_distance_index = list()
            for vidx in value:
                distance = abs(key[0] - vidx[0])
                trial_distance_index.append(distance)
            sorted_value = self.sort_list(value, trial_distance_index)
            trial_result_dict[key] = sorted_value

        return trial_result_dict

    @staticmethod
    def prep_trial(conf_list):
        # generate trails collection from list
        trial_dict = dict()
        for cidx in conf_list:
            trial_dict[cidx] = list()
            for didx in conf_list:
                if cidx != didx:
                    trial_dict[cidx].append(didx)
        return trial_dict

    def switcher(self, conf_id):
        if conf_id == 0:
            return self.model_list
        elif conf_id == 1:
            return self.batch_size_list
        elif conf_id == 2:
            return self.opt_list
        elif conf_id == 3:
            return self.learn_rate_list
        elif conf_id == 4:
            return self.activation_list

    def compute_euclid(self, conf_a, conf_b):
        # compute the euclid distance of two configurations
        pair_euclid_distance = 0
        for cfi, cfv in enumerate(conf_a):
            if isinstance(cfv, str):
                if '-' in cfv:
                    if conf_a[cfi].split('-')[0] == conf_b[cfi].split('-')[0]:
                        conf_list = self.model_layer_list
                        conf_a_conf_idx = conf_list.index(int(cfv.split('-')[1]))
                        conf_b_conf_idx = conf_list.index(int(conf_b[cfi].split('-')[1]))
                        pair_euclid_distance += abs(conf_a_conf_idx - conf_b_conf_idx)
                elif conf_a[cfi] != conf_b[cfi]:
                    pair_euclid_distance += 1
            else:
                conf_list = self.switcher(cfi)
                conf_a_conf_idx = conf_list.index(cfv)
                conf_b_conf_idx = conf_list.index(conf_b[cfi])
                pair_euclid_distance += abs(conf_a_conf_idx - conf_b_conf_idx)
        return pair_euclid_distance

    def sort_conf_euclid(self, trial_dict):
        # sort the configurations according to euclid distance
        trial_result_dict = dict()
        for key, value in trial_dict.items():
            trial_result_dict[key] = list()
            trial_distance_index = list()
            for vidx in value:
                distance = self.compute_euclid(key, vidx)
                trial_distance_index.append(distance)
            sorted_value = self.sort_list(value, trial_distance_index)
            trial_result_dict[key] = sorted_value
        return trial_result_dict

    def knn_conf_euclid(self, confs, topk):
        # schedule confs using euclid-based knn
        confs_list = list(confs)
        print(confs_list)
        trial_dict = self.prep_trial(confs_list)
        trial_result_dict = self.sort_conf_euclid(trial_dict)

        trial_pack_collection = []

        while len(confs_list) > 0:
            trial_packed_list = []
            spoint = rd.choice(confs_list)
            trial_packed_list.append(spoint)
            ssl = trial_result_dict.get(spoint)

            if topk <= len(ssl):
                for stidx in range(topk):
                    selected_conf = ssl[stidx]
                    # print("selected_conf:",selected_conf)
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
                    # print("selected_conf:",selected_conf)
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

    def knn_conf_bs(self, confs, topk):
        # schedule confs using batchsize-based knn
        confs_list = list(confs)
        trial_dict = self.prep_trial(confs_list)
        trial_result_dict = self.sort_conf_bs(trial_dict)

        trial_pack_collection = []

        while len(confs_list) > 0:
            trial_packed_list = []
            spoint = rd.choice(confs_list)
            trial_packed_list.append(spoint)

            ssl = trial_result_dict.get(spoint)
            # ssl = sorted(spoint_list.items(), key=lambda kv: kv[1], reverse=True)

            if topk <= len(ssl):
                for stidx in range(topk):
                    selected_conf = ssl[stidx]
                    # print("selected_conf:",selected_conf)
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
                    print("selected_conf:{}".format(selected_conf))
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

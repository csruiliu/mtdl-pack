import random as rd
import config.config_parameter as cfg_para

##################################################################################
# euclid-based KNN
##################################################################################

switcher = {0: 'model_arch_global',
            1: 'batch_size_global',
            2: 'opt_conf_global',
            3: 'learning_rate_global',
            4: 'activation_global'}

model_arch_global = cfg_para.hyperband_model_type_list
model_layer_global = list()
for model_arch in model_arch_global:
    model_layer_global.append(int(model_arch.split('-')[1]))
batch_size_global = cfg_para.hyperband_batch_size_list
opt_global = cfg_para.hyperband_optimizer_list
learning_rate_global = cfg_para.hyperband_learn_rate_list
activation_global = cfg_para.hyperband_activation_list

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


# compute the euclid distance of two configurations
def compute_euclid(conf_a, conf_b):
    pair_euclid_distance = 0
    for cfi, cfv in enumerate(conf_a):
        if isinstance(cfv, str):
            if '-' in cfv:
                if conf_a[cfi].split('-')[0] == conf_b[cfi].split('-')[0]:
                    conf_list = list(globals()['model_layer_global'])
                    conf_a_conf_idx = conf_list.index(cfv)
                    conf_b_conf_idx = conf_list.index(conf_b[cfi])
                    pair_euclid_distance += abs(conf_a_conf_idx - conf_b_conf_idx)
            elif conf_a[cfi] != conf_b[cfi]:
                pair_euclid_distance += 1
        else:
            conf_list = list(globals()[switcher.get(cfi)])
            conf_a_conf_idx = conf_list.index(cfv)
            conf_b_conf_idx = conf_list.index(conf_b[cfi])
            pair_euclid_distance += abs(conf_a_conf_idx - conf_b_conf_idx)
    return pair_euclid_distance


# generate trails collection from list
def prep_trial(conf_list):
    trial_dict = dict()
    for cidx in conf_list:
        trial_dict[cidx] = list()
        for didx in conf_list:
            if cidx != didx:
                trial_dict[cidx].append(didx)
    return trial_dict


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

##################################################################################
# batch size-based KNN
##################################################################################

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


# schedule confs using batchsize-based knn
def knn_conf_bs(confs, topk):
    confs_list = list(confs)
    trial_dict = prep_trial(confs_list)
    trial_result_dict = sort_conf_bs(trial_dict)

    trial_pack_collection = []

    while len(confs_list) > 0:
        trial_packed_list = []
        spoint = rd.choice(confs_list)
        # print("spoint:",spoint)
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

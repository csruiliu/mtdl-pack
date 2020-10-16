import yaml
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

with open(current_folder+'/config_parameter.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

##########################################
# Hyperparameters for input data
##########################################

hyperparams_input = cfg['hyperparams_input']
img_width_imagenet = hyperparams_input['img_width_imagenet']
img_height_imagenet = hyperparams_input['img_height_imagenet']
img_width_cifar10 = hyperparams_input['img_width_cifar10']
img_height_cifar10 = hyperparams_input['img_height_cifar10']
img_width_mnist = hyperparams_input['img_width_mnist']
img_height_mnist = hyperparams_input['img_height_mnist']
num_class_imagenet = hyperparams_input['num_class_imagenet']
num_class_cifar10 = hyperparams_input['num_class_cifar10']
num_class_mnist = hyperparams_input['num_class_mnist']
num_channels_rgb = hyperparams_input['num_channel_rgb']
num_channels_bw = hyperparams_input['num_channel_bw']

##########################################
# Hyperparameters for single training
##########################################

hyperparams_single_cfg = cfg['hyperparams_single_train']
single_num_epoch = hyperparams_single_cfg['num_epoch']
single_rand_seed = hyperparams_single_cfg['random_seed']
single_model_type = hyperparams_single_cfg['model_type']
single_activation = hyperparams_single_cfg['activation']
single_opt = hyperparams_single_cfg['optimizer']
single_batch_size = hyperparams_single_cfg['batch_size']
single_num_layer = hyperparams_single_cfg['num_model_layer']
single_learning_rate = hyperparams_single_cfg['learning_rate']
single_train_dataset = hyperparams_single_cfg['train_dataset']
single_use_cpu = hyperparams_single_cfg['use_cpu']
single_use_tb_timeline = hyperparams_single_cfg['use_tb_timeline']

##########################################
# Hyperparameters for multiple training
##########################################

hyperparams_multi_cfg = cfg['hyperparams_multiple_train']
multi_rand_seed = hyperparams_multi_cfg['random_seed']
multi_num_epoch = hyperparams_multi_cfg['num_epoch']
multi_model_type = hyperparams_multi_cfg['model_type']
multi_activation = hyperparams_multi_cfg['activation']
multi_opt = hyperparams_multi_cfg['optimizer']
multi_batch_size = hyperparams_multi_cfg['batch_size']
multi_num_layer = hyperparams_multi_cfg['num_model_layer']
multi_learning_rate = hyperparams_multi_cfg['learning_rate']
multi_train_dataset = hyperparams_multi_cfg['train_dataset']
multi_batch_padding = hyperparams_multi_cfg['batch_padding']
multi_use_cpu = hyperparams_multi_cfg['use_cpu']
multi_same_input = hyperparams_multi_cfg['same_input']
multi_use_tb_timeline = hyperparams_multi_cfg['use_tb_timeline']

##########################################
# Hyperparameters for hyperband
##########################################

hyperparams_hyperband = cfg['hyperparams_hyperband']
hyperband_resource_conf = hyperparams_hyperband['resource_conf']
hyperband_down_rate = hyperparams_hyperband['down_rate']
hyperband_pack_rate = hyperparams_hyperband['pack_rate']
hyperband_schedule_policy = hyperparams_hyperband['schedule_policy']
hyperband_random_seed = hyperparams_hyperband['workload_random_seed']
hyperband_model_type_list = hyperparams_hyperband['model_type']
hyperband_activation_list = hyperparams_hyperband['activation']
hyperband_optimizer_list = hyperparams_hyperband['optimizer']
hyperband_batch_size_list = hyperparams_hyperband['batch_size']
hyperband_learn_rate_list = hyperparams_hyperband['learning_rate']

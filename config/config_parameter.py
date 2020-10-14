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
# Hyperparameters for pack training
##########################################

hyperparams_pack_cfg = cfg['hyperparams_pack_train']
pack_num_epoch = hyperparams_pack_cfg['num_epoch']
pack_rand_seed = hyperparams_pack_cfg['random_seed']
pack_model_type = hyperparams_pack_cfg['packed_model_type']
pack_activation = hyperparams_pack_cfg['activation']
pack_opt = hyperparams_pack_cfg['optimizer']
pack_batch_size = hyperparams_pack_cfg['batch_size']
pack_num_layer = hyperparams_pack_cfg['num_model_layer']
pack_learning_rate = hyperparams_pack_cfg['learning_rate']
pack_train_dataset = hyperparams_pack_cfg['train_dataset']
pack_batch_padding = hyperparams_pack_cfg['batch_padding']
pack_use_cpu = hyperparams_pack_cfg['use_cpu']
pack_same_input = hyperparams_pack_cfg['same_input']
pack_use_tb_timeline = hyperparams_pack_cfg['use_tb_timeline']


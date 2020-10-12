import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#with open("config_cifar10.yml", 'r') as ymlfile:
#    cfg = yaml.load(ymlfile)

hyperband_config = cfg['hyperband']

resource_conf = hyperband_config['resource_conf']
down_rate = hyperband_config['down_rate']
pack_rate_sch = hyperband_config['pack_rate'] 
sch_policy = hyperband_config['schedule_policy']

hyperparams_cfg = cfg['hypermeter']

imgWidth = hyperparams_cfg['img_width']
imgHeight = hyperparams_cfg['img_height']
batch_size = hyperparams_cfg['batch_size']
opt_conf = hyperparams_cfg['optimizer']
model_layer = hyperparams_cfg['num_model_layer']
activation = hyperparams_cfg['activation']
learning_rate = hyperparams_cfg['learning_rate']
model_type = hyperparams_cfg['model_type']
numChannels = hyperparams_cfg['num_channel']
numClasses = hyperparams_cfg['num_class']
rand_seed = hyperparams_cfg['random_seed']

#data_path_cfg = cfg['local_data_path']
data_path_cfg = cfg['remote_data_path']

mnist_train_img_path = data_path_cfg['mnist_train_img_path']
mnist_train_label_path = data_path_cfg['mnist_train_label_path']
mnist_t10k_img_path = data_path_cfg['mnist_t10k_img_path']
mnist_t10k_label_path = data_path_cfg['mnist_t10k_label_path']
cifar_10_path = data_path_cfg['cifar_10_path']
imagenet_t1k_img_path = data_path_cfg['imagenet_t1k_img_path']
imagenet_t1k_label_path = data_path_cfg['imagenet_t1k_label_path']
imagenet_t1k_bin_path = data_path_cfg['imagenet_t1k_bin_path']
imagenet_t10k_img_path = data_path_cfg['imagenet_t10k_img_path']
imagenet_t10k_label_path = data_path_cfg['imagenet_t10k_label_path']
imagenet_t10k_bin_path = data_path_cfg['imagenet_t10k_bin_path']
           
    

import yaml
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

with open(current_folder+'/config_path.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#path_cfg = cfg['local_path']
path_cfg = cfg['roscoe_path']
#path_cfg = cfg['hp_path']

imagenet_t50k_img_raw_path = path_cfg['dataset_imagenet_t50k_img_raw']
imagenet_t50k_label_path = path_cfg['dataset_imagenet_t50k_label']
imagenet_t10k_img_raw_path = path_cfg['dataset_imagenet_t10k_img_raw']
imagenet_t10k_img_bin_path = path_cfg['dataset_imagenet_t10k_img_bin']
imagenet_t10k_label_path = path_cfg['dataset_imagenet_t10k_label']
imagenet_t1k_img_raw_path = path_cfg['dataset_imagenet_t1k_img_raw']
imagenet_t1k_img_bin_path = path_cfg['dataset_imagenet_t1k_img_bin']
imagenet_t1k_label_path = path_cfg['dataset_imagenet_t1k_label']
mnist_train_img_path = path_cfg['dataset_mnist_train_img']
mnist_train_label_path = path_cfg['dataset_mnist_train_label']
mnist_test_10k_img_path = path_cfg['dataset_mnist_t10k_img']
mnist_test_10k_label_path = path_cfg['dataset_mnist_t10k_label']
cifar_10_path = path_cfg['dataset_cifar_10_path']

profile_path = path_cfg['profile_path']

import pack.config.config_parameter as cfg_para
import pack.config.config_path as cfg_path
from pack.tools.img_tool import load_imagenet_labels_onehot
from pack.tools.img_tool import load_cifar10_keras
from pack.tools.img_tool import load_mnist_image, load_mnist_label_onehot


def data_loader(dataset_arg):
    if dataset_arg == 'imagenet':
        img_width = cfg_para.img_width_imagenet
        img_height = cfg_para.img_height_imagenet
        num_channel = cfg_para.num_channels_rgb
        num_class = cfg_para.num_class_imagenet

        train_feature = cfg_path.imagenet_t50k_img_raw_path
        test_feature = cfg_path.imagenet_t1k_img_raw_path

        train_label_path = cfg_path.imagenet_t50k_label_path
        test_label_path = cfg_path.imagenet_t1k_label_path
        train_label = load_imagenet_labels_onehot(train_label_path, num_class)
        test_label = load_imagenet_labels_onehot(test_label_path, num_class)

    elif dataset_arg == 'cifar10':
        img_width = cfg_para.img_width_cifar10
        img_height = cfg_para.img_height_cifar10
        num_channel = cfg_para.num_channels_rgb
        num_class = cfg_para.num_class_cifar10

        train_feature, train_label, test_feature, test_label = load_cifar10_keras()

    elif dataset_arg == 'mnist':
        img_width = cfg_para.img_width_mnist
        img_height = cfg_para.img_height_mnist
        num_channel = cfg_para.num_channels_bw
        num_class = cfg_para.num_class_mnist

        train_img_path = cfg_path.mnist_train_img_path
        train_label_path = cfg_path.mnist_train_label_path
        test_img_path = cfg_path.mnist_test_10k_img_path
        test_label_path = cfg_path.mnist_test_10k_label_path

        train_feature = load_mnist_image(train_img_path)
        train_label = load_mnist_image(train_label_path)
        test_feature = load_mnist_image(test_img_path)
        test_label = load_mnist_label_onehot(test_label_path)

    else:
        raise ValueError('Training Dataset is invaild, only support mnist, cifar10, imagenet')

    return [img_width, img_height, num_channel, num_class, train_feature, train_label, test_feature, test_label]
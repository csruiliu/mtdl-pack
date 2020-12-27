import numpy as np
import pickle
import cv2
import os
from keras.datasets import cifar10
from keras.utils import to_categorical


#####################################
# read mnist training data
#####################################
def load_mnist_image(path):
    mnist_num_channels = 1
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_img_width = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_img_height = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(mnist_img_width * mnist_img_height * num_images)
        img_raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        img = img_raw.reshape(num_images, mnist_img_height, mnist_img_width, mnist_num_channels)

    return img


#####################################
# read mnist test data
#####################################
def load_mnist_label_onehot(path):
    num_classes = 10
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        labels_array = np.zeros((num_images, num_classes))
        for lidx, lval in enumerate(labels):
            labels_array[lidx, lval] = 1

    return labels_array


########################################################
# read cifar-10 data using keras
########################################################
def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_test


def load_cifar10_keras():
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    return train_data, train_labels, test_data, test_labels


########################################################
# read cifar-10 data, batch 1-5 training data
########################################################
def load_cifar10_train(path):

    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []
    cifar_label_train_onehot = np.zeros((50000, 10))

    for i in range(1, 6):
        with open(path + '/data_batch_' + str(i), 'rb') as fo:
            data_batch = pickle.load(fo, encoding='bytes')
            if i == 1:
                cifar_train_data = data_batch[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, data_batch[b'data']))
            cifar_train_filenames += data_batch[b'filenames']
            cifar_train_labels += data_batch[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    cifar_train_labels = np.array(cifar_train_labels)

    for cl in range(50000):
        cifar_label_train_onehot[cl, cifar_train_labels[cl]] = 1

    return cifar_train_data, cifar_label_train_onehot


########################################################
# read cifar-10 data, testing data
########################################################
def load_cifar10_eval(path):
    with open(path + '/test_batch', 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
        test_data = test_batch[b'data']
        test_label = test_batch[b'labels']

    cifar_test_data = test_data.reshape((len(test_data), 3, 32, 32))
    cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)

    cifar_label_test_onehot = np.zeros((10000, 10))

    for cl in range(10000):
        cifar_label_test_onehot[cl, test_label[cl]] = 1

    return cifar_test_data, cifar_label_test_onehot


########################################################
# read imagenet raw images
########################################################
def load_imagenet_raw(image_dir, batch_list, img_h, img_w):
    img_list = []
    for img in batch_list:
        # print(image_dir+"/"+img)
        im = cv2.imread(image_dir + "/" + img, cv2.IMREAD_COLOR)
        res = cv2.resize(im, dsize=(img_w, img_h))
        res_exp = np.expand_dims(res, axis=0)
        img_list.append(res_exp)
    img_data = np.concatenate(img_list, axis=0)
    return img_data


########################################################
# read imagenet label
########################################################
def load_imagenet_labels_onehot(path):
    num_classes = 1000
    lines = open(path).readlines()
    labels_array = np.zeros((len(lines), num_classes))
    for idx, val in enumerate(lines):
        hot = int(val.rstrip('\n'))
        labels_array[idx, hot - 1] = 1
    return labels_array


########################################################
# read imagenet bin
########################################################
def load_imagenet_bin(path, num_channels, img_w, img_h):
    image_arr = np.fromfile(path, dtype=np.uint8)
    img_num = int(image_arr.size / img_w / img_h / num_channels)
    images = image_arr.reshape((img_num, img_w, img_h, num_channels))
    return images


########################################################
# convert imagenet raw images to bin
########################################################
def convert_imagenet_bin(path):
    img_w = 224
    img_h = 224
    img_list = []
    img_filename_list = sorted(os.listdir(path))
    for filename in img_filename_list:
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            img_resize = cv2.resize(img, dsize=(img_w, img_h))
            img_expand = np.expand_dims(img_resize, axis=0)
            img_list.append(img_expand)
    img_data = np.concatenate(img_list, axis=0)
    output_file = open("imagenet1k.bin", "wb")
    binary_format = bytearray(img_data)
    output_file.write(binary_format)
    output_file.close()
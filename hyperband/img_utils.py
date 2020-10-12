import numpy as np
import pickle
import cv2


#####################################
# read mnist training data
#####################################
def load_mnist_image(path, rd_seed):
    mnist_numChannels = 1
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_imgWidth = int.from_bytes(bytestream.read(4), byteorder='big')
        mnist_imgHeight = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(mnist_imgWidth * mnist_imgHeight * num_images)
        img_raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        img = img_raw.reshape(num_images, mnist_imgHeight, mnist_imgWidth, mnist_numChannels)
        np.random.seed(rd_seed)
        np.random.shuffle(img)
    return img


#####################################
# read mnist test data
#####################################
def load_mnist_label_onehot(path, rd_seed):
    num_classes = 10
    with open(path, 'rb') as bytestream:
        _ = int.from_bytes(bytestream.read(4), byteorder='big')
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        labels_array = np.zeros((num_images, num_classes))
        for lidx, lval in enumerate(labels):
            labels_array[lidx, lval] = 1
    np.random.seed(rd_seed)
    np.random.shuffle(labels_array)
    return labels_array


########################################################
# read cifar-10 data, batch 1-5 training data
########################################################
def load_cifar_train(path, rd_seed):
    cifar_data_train = []
    cifar_label_train = []
    cifar_label_train_onehot = np.zeros((50000, 10))

    for i in range(1, 6):
        with open(path + '/data_batch_' + str(i), 'rb') as fo:
            data_batch = pickle.load(fo, encoding='bytes')
            train_data = data_batch[b'data']
            label_data = data_batch[b'labels']

        if cifar_data_train == []:
            cifar_data_train = train_data
        else:
            cifar_data_train = np.concatenate((cifar_data_train, train_data))

        cifar_label_train = cifar_label_train + label_data

    cifar_train = cifar_data_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')

    for cl in range(50000):
        cifar_label_train_onehot[cl, cifar_label_train[cl]] = 1

    np.random.seed(rd_seed)
    np.random.shuffle(cifar_train)
    np.random.shuffle(cifar_label_train_onehot)

    return cifar_train, cifar_label_train_onehot


########################################################
# read cifar-10 data, testing data
########################################################
def load_cifar_test(path, rd_seed):
    with open(path + '/test_batch', 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
        test_data = test_batch[b'data']
        label_data = test_batch[b'labels']

    cifar_data_test = test_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')
    cifar_label_test_onehot = np.zeros((10000, 10))

    for cl in range(10000):
        cifar_label_test_onehot[cl, label_data[cl]] = 1

    np.random.seed(rd_seed)
    np.random.shuffle(cifar_data_test)
    np.random.shuffle(cifar_label_test_onehot)

    return cifar_data_test, cifar_label_test_onehot


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
def load_imagenet_labels_onehot(path, num_classes):
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
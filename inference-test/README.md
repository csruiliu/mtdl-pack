# TensorFlow Experiments #

This is the preliminary experiments for multitenancy of deep learning

1. Single test: test single image inference using MobileNet and ResNet and sequential models

2. Batch test: we build a packed model (ResNet and MobileNet) and test it using 10000 images from ImageNet with batch size 1000, 100, 10. We also use MobileNet and ResNet as benchmarks

## Run experiments ##

1. Clone tensorflow model: git clone https://github.com/tensorflow/models.git in the root folder of this project.

2. Download checkponts: ResNet V2 50: http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz, MoblieNet V2 224: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz 

3. Put 10000 images to data folder

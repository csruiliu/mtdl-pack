import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.keras.layers import GlobalAveragePooling2D

growth_k = 32

#DenseNet-121
class densenet(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.0001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.num_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding
        self.model_logit = None
        self.train_op = None
        self.eval_op = None
        self.num_conv_layer = 0
        self.num_pool_layer = 0
        self.num_dense_layer = 0


    def bottleneck_block(self, input, block_name):
        with tf.variable_scope(block_name):
            block = tf.layers.batch_normalization(input, training=True, trainable=True, name=block_name+'_bn_0')
            block = tf.nn.relu(block, name=block_name+'_relu_0')
            block = tf.layers.conv2d(block, filters=2*growth_k, kernel_size=(1, 1), strides=1, padding='same', name=block_name+'_conv_0')
            self.num_conv_layer += 1

            block = tf.layers.batch_normalization(block, training=True, trainable=True, name=block_name+'_bn_1')
            block = tf.nn.relu(block, name=block_name+'_relu_1')
            block = tf.layers.conv2d(block, filters=2*growth_k, kernel_size=(3, 3), strides=1, padding='same', name=block_name+'_conv_1')
            self.num_conv_layer += 1

            return block

    def dense_block(self, input, dn_layers, block_name):
        with tf.variable_scope(block_name): 
            block = self.bottleneck_block(input, block_name=block_name+'_bottleneck_block_0')
            block_input = tf.concat(values=[input, block], axis=3)
            for i in range(dn_layers - 1):
                block = self.bottleneck_block(block_input, block_name=block_name+'_bottleneck_block_'+str(i+1))
                block_input = tf.concat([block_input,block], axis=3)
            
            return block
    
    def transition_block(self, input, block_name):
        with tf.variable_scope(block_name):
            block = tf.layers.batch_normalization(input, training=True, trainable=True, name=block_name+'_bn_0')
            block = tf.layers.conv2d(block, filters=2*growth_k, kernel_size=(1,1), strides=1, padding='same', name=block_name+'_conv_0')
            self.num_conv_layer += 1
            block = tf.layers.average_pooling2d(block, pool_size=(2,2), strides=2, padding='same', name=block_name+'_avgpool_0')
            self.num_pool_layer += 1
            return block

    def build(self, input):
        if self.batch_padding == True:
            input = input[0:self.batch_size, :, :, :]

        with tf.variable_scope(self.net_name + '_instance'):
            net = tf.layers.conv2d(input, filters=2*growth_k, kernel_size=(7,7), strides=2, padding='same', name='conv_0')
            self.num_conv_layer += 1
            net = tf.layers.max_pooling2d(net, pool_size=(3,3), strides=2, padding='same', name='max_pool_0')
            self.num_pool_layer += 1
            net = self.dense_block(net, dn_layers=6, block_name='dense_block_0')
            net = self.transition_block(net, block_name='trans_block_0')
            net = self.dense_block(net, dn_layers=12, block_name='dense_block_1')
            net = self.transition_block(net, block_name='trans_block_1')
            net = self.dense_block(net, dn_layers=24, block_name='dense_block_2')
            net = self.transition_block(net, block_name='trans_block_2')
            net = self.dense_block(net, dn_layers=16, block_name='dense_block_3')
            net = GlobalAveragePooling2D()(net)
            self.num_pool_layer += 1
            net = tc.layers.flatten(net)
            self.num_dense_layer += 1
            self.model_logit = tf.layers.dense(net, units=self.num_classes, name='full_connected')
            self.num_dense_layer += 1

        print('================number of conv layer===================:', self.num_conv_layer)
        print('================number of pool layer===================:', self.num_pool_layer)
        print('================number of dense layer===================:', self.num_dense_layer)
        return self.model_logit
    
    def train(self, logits, labels):
        if self.batch_padding == True:
            labels = labels[0:self.batch_size, :]

        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.opt == 'Adam':
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.opt == 'SGD':
                    self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.opt == 'Adagrad':
                    self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.opt == 'Momentum':
                    self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(cross_entropy_cost)

        return self.train_op

    def evaluate(self, logits, labels):
        with tf.name_scope('eval_' + self.net_name):
            pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            self.eval_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return self.eval_op

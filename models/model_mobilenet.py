import tensorflow as tf
import tensorflow.contrib as tc

# MobileNetV2
class mobilenet(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.0001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.num_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.num_classes = num_classes
        self.batch_size = tf.Variable(batch_size)
        self.opt = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding

        self.model_logit = None
        self.train_op = None
        self.eval_op = None

        self.num_conv_layer = 0
        self.num_pool_layer = 0
        self.num_residual_layer = 0

    @staticmethod
    def activation_layer(x_input, act_func):
        new_logit = None
        if act_func == 'relu':
            new_logit = tf.nn.relu(x_input, 'relu')
        elif act_func == 'leaky_relu':
            new_logit = tf.nn.leaky_relu(x_input, alpha=0.2, name='leaky_relu')
        elif act_func == 'tanh':
            new_logit = tf.math.tanh(x_input, 'tanh')
        elif act_func == 'sigmoid':
            new_logit = tf.math.sigmoid(x_input, 'sigmoid')
        elif act_func == 'elu':
            new_logit = tf.nn.elu(x_input, 'elu')
        elif act_func == 'selu':
            new_logit = tf.nn.selu(x_input, 'selu')

        return new_logit

    def _res_block(self, input, expansion_ratio, output_dim, stride, is_train, block_name, bias=False, shortcut=True):
        with tf.variable_scope(block_name):
            bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
            net = self._conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
            net = self._batch_norm(net, train=is_train, name='pw_bn')
            net = self.activation_layer(net, self.activation)

            net = self._dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = self._batch_norm(net, train=is_train, name='dw_bn')
            net = self.activation_layer(net, self.activation)

            net = self._conv_1x1(net, output_dim, name='pw_linear', bias=bias)
            net = self._batch_norm(net, train=is_train, name='pw_linear_bn')

            # element wise add, only for stride==1
            if shortcut and stride == 1:
                in_dim = int(input.get_shape().as_list()[-1])
                if in_dim != output_dim:
                    ins = self._conv_1x1(input, output_dim, name='ex_dim')
                    net = ins + net
                    self.add_layer_num('residual', 1)
                else:
                    net = input + net
                    self.add_layer_num('residual', 1)

        return net

    def _conv2d_block(self, x_init, out_dim, kernel_size, strides_size, is_train, block_name):
        with tf.variable_scope(block_name):
            block = self._conv2d(x_init, out_dim, kernel_size, kernel_size, strides_size, strides_size)
            block = self._batch_norm(block, train=is_train, name='bn')
            block = self.activation_layer(block, self.activation)
        return block

    def _pwise_block(self, x_init, output_dim, is_train, block_name, bias=False):
        with tf.variable_scope(block_name):
            out = self._conv_1x1(x_init, output_dim, bias=bias, name='pwb')
            out = self._batch_norm(out, train=is_train, name='bn')
            block = self.activation_layer(out, self.activation)
        return block

    def _conv2d(self, x_init, output_dim, kernel_height, kernel_width, strides_h, strides_w, stddev=0.02, bias=False,
                name='conv2d'):
        weight_decay = 1e-4
        with tf.variable_scope(name):
            w = tf.get_variable('w', [kernel_height, kernel_width, x_init.get_shape()[-1], output_dim],
                                regularizer=tc.layers.l2_regularizer(weight_decay),
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(x_init, w, strides=[1, strides_h, strides_w, 1], padding='SAME')
            self.add_layer_num('conv', 1)
            if bias:
                biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
        return conv

    def _dwise_conv(self, x_init, kernel_height=3, kernel_width=3, channel_multiplier=1, strides=[1, 1, 1, 1],
                    padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
        weight_decay = 1e-4
        with tf.variable_scope(name):
            in_channel = x_init.get_shape().as_list()[-1]
            w = tf.get_variable('w', [kernel_height, kernel_width, in_channel, channel_multiplier],
                                regularizer=tc.layers.l2_regularizer(weight_decay),
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.depthwise_conv2d(x_init, w, strides, padding, rate=None, name=None, data_format=None)
            self.add_layer_num('conv', 1)
            if bias:
                biases = tf.get_variable('bias', [in_channel * channel_multiplier],
                                         initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
        return conv

    def _batch_norm(self, x_init, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
        return tf.layers.batch_normalization(x_init, momentum=momentum, epsilon=epsilon, scale=True, training=train,
                                             name=name)

    def _conv_1x1(self, x_init, output_dim, name, bias=False):
        with tf.variable_scope(name):
            net = self._conv2d(x_init, output_dim, 1, 1, 1, 1, stddev=0.02, name=name, bias=bias)
            self.add_layer_num('conv', 1)
        return net

    def _global_avg(self, x):
        with tf.name_scope('global_avg'):
            net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
            self.add_layer_num('pool', 1)
        return net

    def build(self, input_features, is_training):
        exp = 6
        if self.batch_padding:
            train_input = input_features[0:self.batch_size, :, :, :]
        else:
            train_input = input_features

        with tf.variable_scope(self.net_name + '_instance'):
            net = self._conv2d_block(train_input, 32, 3, 2, is_training, 'conv1_1')
            net = self._res_block(net, 1, 16, 1, is_training, block_name='res2_1')
            net = self._res_block(net, exp, 24, 2, is_training, block_name='res3_1')  # size/4
            net = self._res_block(net, exp, 24, 1, is_training, block_name='res3_2')

            net = self._res_block(net, exp, 32, 2, is_training, block_name='res4_1')  # size/8
            net = self._res_block(net, exp, 32, 1, is_training, block_name='res4_2')
            net = self._res_block(net, exp, 32, 1, is_training, block_name='res4_3')

            net = self._res_block(net, exp, 64, 2, is_training, block_name='res5_1')
            net = self._res_block(net, exp, 64, 1, is_training, block_name='res5_2')
            net = self._res_block(net, exp, 64, 1, is_training, block_name='res5_3')
            net = self._res_block(net, exp, 64, 1, is_training, block_name='res5_4')

            net = self._res_block(net, exp, 96, 1, is_training, block_name='res6_1')  # size/16
            net = self._res_block(net, exp, 96, 1, is_training, block_name='res6_2')
            net = self._res_block(net, exp, 96, 1, is_training, block_name='res6_3')

            net = self._res_block(net, exp, 160, 2, is_training, block_name='res7_1')  # size/32
            net = self._res_block(net, exp, 160, 1, is_training, block_name='res7_2')
            net = self._res_block(net, exp, 160, 1, is_training, block_name='res7_3')

            net = self._res_block(net, exp, 320, 1, is_training, block_name='res8_1', shortcut=False)
            net = self._pwise_block(net, 1280, is_training, block_name='conv9_1')
            net = self._global_avg(net)

            self.model_logit = tc.layers.flatten(self._conv_1x1(net, self.num_classes, name='logits'))

        return self.model_logit

    def train(self, logits, train_labels):
        if self.batch_padding:
            batch_labels = train_labels[0:self.batch_size]
        else:
            batch_labels = train_labels

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch_labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        reg_loss = tf.losses.get_regularization_loss()
        train_loss = cross_entropy_cost + reg_loss

        if self.opt == 'Adam':
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(train_loss)
        elif self.opt == 'SGD':
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(train_loss)
        elif self.opt == 'Adagrad':
            self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(train_loss)
        elif self.opt == 'Momentum':
            self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(train_loss)

        return self.train_op

    def evaluate(self, logits, eval_labels):
        prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(eval_labels, -1))
        self.eval_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return self.eval_op

    def add_layer_num(self, layer_type, layer_num):
        if layer_type == 'pool':
            self.num_pool_layer += layer_num
        elif layer_type == 'conv':
            self.num_conv_layer += layer_num
        elif layer_type == 'residual':
            self.num_residual_layer += layer_num

    def set_batch_size(self, batch_size):
        return self.batch_size.assign(batch_size)

    def get_layer_info(self):
        return self.num_conv_layer, self.num_pool_layer, self.num_residual_layer

    def print_model_info(self):
        print('=====================================================================')
        print('number of conv layer: {}, number of pooling layer: {}, number of residual layer: {}'
              .format(self.num_conv_layer, self.num_pool_layer, self.num_residual_layer))
        print('=====================================================================')

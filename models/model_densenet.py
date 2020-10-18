import tensorflow as tf
import tensorflow.contrib as tc


# DenseNet
class densenet(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.0001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.residual_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.num_classes = num_classes
        self.batch_size = tf.Variable(batch_size)
        self.opt = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding
        self.growth_k = 32

        self.residual_layer_list = list()
        self.model_logit = None
        self.train_op = None
        self.eval_op = None

        self.num_conv_layer = 0
        self.num_pool_layer = 0
        self.num_residual_layer = 0

        self.weight_init = tc.layers.variance_scaling_initializer()
        self.weight_regularizer = tc.layers.l2_regularizer(0.0001)

        self.cur_step = 1
        self.cur_epoch = 1
        self.desire_steps = -1
        self.desire_epochs = -1

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

    def avg_pooling_layer(self, x_input, pool_size, scope='avgpool'):
        self.add_layer_num('pool', 1)
        return tf.layers.average_pooling2d(x_input, pool_size=pool_size, strides=2, padding='same', name=scope)

    def max_pooling_layer(self, x_input, pool_size, scope='maxpool'):
        self.add_layer_num('pool', 1)
        return tf.layers.max_pooling2d(x_input, pool_size=pool_size, strides=2, padding='same', name=scope)

    def global_avg_pooling(self, x_input):
        self.add_layer_num('pool', 1)
        gap = tf.reduce_mean(x_input, axis=[1, 2], keepdims=True)
        return gap

    def conv_layer(self, x_input, filters, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
        self.add_layer_num('conv', 1)
        with tf.variable_scope(scope):
            layer = tf.layers.conv2d(inputs=x_input, filters=filters, kernel_size=kernel,
                                     kernel_initializer=self.weight_init, strides=stride, use_bias=use_bias,
                                     padding=padding)
            return layer

    def fully_conneted_layer(self, x_input, units, use_bias=True, scope='fully_0'):
        with tf.variable_scope(scope):
            layer = tf.layers.flatten(x_input)
            layer = tf.layers.dense(layer, units=units, kernel_initializer=self.weight_init,
                                    kernel_regularizer=self.weight_regularizer, use_bias=use_bias)
            return layer

    def batch_norm_layer(self, x_input, is_training=True, scope='batch_norm'):
        return tc.layers.batch_norm(x_input, decay=0.9, epsilon=1e-05, center=True, scale=True,
                                    updates_collections=None, is_training=is_training, scope=scope)

    def bottle_dense_block(self, x_init, is_training=True, use_bias=True, scope='bottle_denseblock'):
        with tf.variable_scope(scope):
            block = self.batch_norm_layer(x_init, is_training, scope='batch_norm_0')
            block = self.activation_layer(block, self.activation)
            block = self.conv_layer(block, filters=self.growth_k, kernel=1, stride=1, use_bias=use_bias, scope='conv_0')

            block = self.batch_norm_layer(block, is_training, scope='batch_norm_1')
            block = self.activation_layer(block, self.activation)
            block = self.conv_layer(block, filters=self.growth_k, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

            return block

    def dense_block(self, x_init, dn_layers, is_training=True, use_bias=True, scope='denseblock'):
        with tf.variable_scope(scope):
            block = self.batch_norm_layer(x_init, is_training, scope='batch_norm_0')
            self.add_layer_num('residual', 1)
            block_input = tf.concat(values=[x_init, block], axis=3)
            for i in range(dn_layers - 1):
                block = self.bottle_dense_block(block_input, is_training, use_bias, scope='bottle_denseblock_' + str(i))
                self.add_layer_num('residual', 1)
                block_input = tf.concat([block_input, block], axis=3)

            return block

    def transition_block(self, x_init, is_training=True, use_bias=True, scope='transblock'):
        with tf.variable_scope(scope):
            block = self.batch_norm_layer(x_init, is_training, scope='batch_norm_0')
            block = self.conv_layer(block, filters=self.growth_k, kernel=1, stride=1, use_bias=use_bias, scope='conv_0')
            block = self.avg_pooling_layer(block, pool_size=2, scope='pool_0')

            return block

    def build(self, input_features, is_training=True):
        if self.batch_padding:
            train_input = input_features[0:self.batch_size]
        else:
            train_input = input_features

        with tf.variable_scope(self.net_name + '_instance'):
            self.get_residual_layer()

            x = self.conv_layer(train_input, filters=self.growth_k, kernel=7, stride=2, scope='conv_0')
            x = self.max_pooling_layer(x, pool_size=3, scope='maxpool_0')

            for lidx, lnum in enumerate(self.residual_layer_list):
                x = self.dense_block(x, dn_layers=lnum, is_training=is_training,
                                     use_bias=True, scope='denseblock_'+str(lidx))
                x = self.transition_block(x, scope='transblock_'+str(lidx))

            x = self.global_avg_pooling(x)
            self.model_logit = self.fully_conneted_layer(x, units=self.num_classes, scope='logit')

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

    def get_residual_layer(self):
        if self.residual_layer == 121:
            self.residual_layer_list = [6, 12, 24, 16]
        elif self.residual_layer == 169:
            self.residual_layer_list = [6, 12, 32, 32]
        elif self.residual_layer == 201:
            self.residual_layer_list = [6, 12, 48, 32]
        elif self.residual_layer == 264:
            self.residual_layer_list = [6, 12, 64, 48]
        else:
            raise ValueError('[DenseNet] number of residual layer is invalid, try 121, 169, 201, 264')

    def add_layer_num(self, layer_type, layer_num):
        if layer_type == 'pool':
            self.num_pool_layer += layer_num
        elif layer_type == 'conv':
            self.num_conv_layer += layer_num
        elif layer_type == 'residual':
            self.num_residual_layer += layer_num

    def set_batch_size(self, batch_size):
        return self.batch_size.assign(batch_size)

    def get_current_step(self):
        return self.cur_step

    def set_current_step(self, cur_step=1):
        self.cur_step += cur_step
        if self.cur_step > self.desire_steps:
            self.cur_step = 0
            self.cur_epoch += 1

    def reset_current_step(self):
        self.cur_step = 0

    def get_current_epoch(self):
        return self.cur_epoch

    def set_current_epoch(self, cur_epoch=1):
        self.cur_epoch += cur_epoch

    def reset_current_epoch(self):
        self.cur_epoch = 0

    def get_desire_steps(self):
        return self.desire_steps

    def set_desire_steps(self, desire_steps):
        self.desire_steps = desire_steps

    def get_desire_epochs(self):
        return self.desire_epochs

    def set_desire_epochs(self, desire_epochs):
        self.desire_epochs = desire_epochs

    def get_train_op(self):
        return self.train_op

    def get_model_instance_name(self):
        return self.net_name

    def is_complete_train(self):
        if (self.cur_epoch == self.desire_epochs) and (self.cur_step == self.desire_steps):
            return True
        else:
            return False

    def get_layer_info(self):
        return self.num_conv_layer, self.num_pool_layer, self.num_residual_layer

    def print_model_info(self):
        print('=====================================================================')
        print('number of conv layer: {}, number of pooling layer: {}, number of residual layer: {}'
              .format(self.num_conv_layer, self.num_pool_layer, self.num_residual_layer))
        print('=====================================================================')

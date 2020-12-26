import tensorflow as tf


class mlp(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.model_instance_name = 'mlp-' + str(num_layer) + '-' + net_name
        self.num_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.input_size = input_h * input_w * num_channel
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

        self.cur_step = 1
        self.cur_epoch = 1
        self.desire_steps = -1
        self.desire_epochs = -1

    def add_layer_num(self, layer_type, layer_num):
        if layer_type == 'pool':
            self.num_pool_layer += layer_num
        elif layer_type == 'conv':
            self.num_conv_layer += layer_num
        elif layer_type == 'residual':
            self.num_residual_layer += layer_num

    def perceptron_layer(self, x_init, layer_name):
        with tf.variable_scope(layer_name):
            weights = tf.Variable(tf.random_normal([int(x_init.shape[1]), self.num_classes]))
            biases = tf.Variable(tf.random_normal([self.num_classes]))
            layer = tf.matmul(x_init, weights) + biases

        return layer

    def build(self, input_features, is_training=True):
        if self.batch_padding:
            train_input = input_features[0:self.batch_size]
        else:
            train_input = input_features

        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(train_input, [-1, self.input_size])
            layer = self.perceptron_layer(input_image, 'perct1')
            for i in range(self.num_layer):
                layer = self.perceptron_layer(layer, 'perct'+str(i))

        return layer

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
        return self.model_instance_name

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

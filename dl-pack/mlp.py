import tensorflow as tf

class mlp(object):
    def __init__(self, net_name, num_layer, input_h, input_w, num_channel, num_classes, batch_size, opt,
                 learning_rate=0.0001, activation='relu', batch_padding=False):
        self.net_name = net_name
        self.num_layer = num_layer
        self.img_h = input_h
        self.img_w = input_w
        self.channel_num = num_channel
        self.input_size = input_h * input_w * num_channel
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding
        self.model_logit = None
        self.train_op = None
        self.eval_op = None
        self.is_training = True



    def perceptron_layer(self, input, layer_name):
        with tf.variable_scope(layer_name):
            weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
            biases = tf.Variable(tf.random_normal([self.num_classes]))
            layer = tf.matmul(input, weights) + biases

        return layer

    def build(self, input):
        if self.batch_padding:
            input = input[0:self.batch_size, :, :, :]

        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input, [-1, self.input_size])
            layer = self.perceptron_layer(input_image, 'perct1')
            for i in range(self.num_layer):
                layer = self.perceptron_layer(layer, 'perct'+str(i))

        return layer

    def train(self, logits, labels):
        if self.batch_padding == True:
            labels = labels[0:self.batch_size, :]

        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer_' + self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name + '_instance')
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




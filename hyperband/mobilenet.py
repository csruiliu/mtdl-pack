import tensorflow as tf
import tensorflow.contrib as tc

weight_decay = 1e-4
exp = 6  

#MobileNetV2
class mobilenet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, channel_num, num_classes, batch_size, opt, learning_rate=0.0001, activation='relu', is_training=True):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.batch_size = tf.Variable(batch_size)
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.optimzier = opt
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.learning_rate = learning_rate
        self.activation = activation
        self.bn_params = {'is_training': self.is_training}
        self.cur_step = 1
        self.cur_epoch = 1
        self.desire_steps = -1
        self.desire_epochs = -1
        self.train_op = None
        self.eval_op = None
        self.model_logit = None

    def build(self, input):
        instance_name = self.net_name + '_instance'
        with tf.variable_scope(instance_name):
            input_padding = input[0:self.batch_size,:,:,:]
            net = self._conv2d_block(input_padding, 32, 3, 2, self.is_training, 'conv1_1')
            net = self._res_block(net, 1, 16, 1, self.is_training, block_name='res2_1')
            net = self._res_block(net, exp, 24, 2, self.is_training, block_name='res3_1')  # size/4
            net = self._res_block(net, exp, 24, 1, self.is_training, block_name='res3_2')

            net = self._res_block(net, exp, 32, 2, self.is_training, block_name='res4_1')  # size/8
            net = self._res_block(net, exp, 32, 1, self.is_training, block_name='res4_2')
            net = self._res_block(net, exp, 32, 1, self.is_training, block_name='res4_3')

            net = self._res_block(net, exp, 64, 2, self.is_training, block_name='res5_1')
            net = self._res_block(net, exp, 64, 1, self.is_training, block_name='res5_2')
            net = self._res_block(net, exp, 64, 1, self.is_training, block_name='res5_3')
            net = self._res_block(net, exp, 64, 1, self.is_training, block_name='res5_4')

            net = self._res_block(net, exp, 96, 1, self.is_training, block_name='res6_1')  # size/16
            net = self._res_block(net, exp, 96, 1, self.is_training, block_name='res6_2')
            net = self._res_block(net, exp, 96, 1, self.is_training, block_name='res6_3')

            net = self._res_block(net, exp, 160, 2, self.is_training, block_name='res7_1')  # size/32
            net = self._res_block(net, exp, 160, 1, self.is_training, block_name='res7_2')
            net = self._res_block(net, exp, 160, 1, self.is_training, block_name='res7_3')

            net = self._res_block(net, exp, 320, 1, self.is_training, block_name='res8_1', shortcut=False)
            net = self._pwise_block(net, 1280, self.is_training, block_name='conv9_1')
            net = self._global_avg(net)

            self.model_logit = tc.layers.flatten(self._conv_1x1(net, self.num_classes, name='logits'))

            return self.model_logit

    def _res_block(self, input, expansion_ratio, output_dim, stride, is_train, block_name, bias=False, shortcut=True):
        with tf.variable_scope(block_name):
            bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
            net = self._conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
            net = self._batch_norm(net, train=is_train, name='pw_bn')
            if self.activation == 'sigmoid':
                net = tf.nn.sigmoid(net)
            elif self.activation == 'leaky_relu':
                net = tf.nn.leaky_relu(net)
            elif self.activation == 'tanh':
                net = tf.nn.tanh(net)
            elif self.activation == 'relu':
                net = tf.nn.relu6(net, 'relu6')
            
            net = self._dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = self._batch_norm(net, train=is_train, name='dw_bn')
            if self.activation == 'sigmoid':
                net = tf.nn.sigmoid(net)
            elif self.activation == 'leaky_relu':
                net = tf.nn.leaky_relu(net)
            elif self.activation == 'tanh':
                net = tf.nn.tanh(net)
            elif self.activation == 'relu':
                net = tf.nn.relu6(net, 'relu6')

            net = self._conv_1x1(net, output_dim, name='pw_linear', bias=bias)
            net = self._batch_norm(net, train=is_train, name='pw_linear_bn')

            # element wise add, only for stride==1
            if shortcut and stride == 1:
                in_dim=int(input.get_shape().as_list()[-1])
                if in_dim != output_dim:
                    ins = self._conv_1x1(input, output_dim, name='ex_dim')
                    net = ins + net
                else:
                    net = input+net
            
            return net
            
    def _conv2d_block(self, input, out_dim, kernel_size, strides_size, is_train, block_name):
        with tf.variable_scope(block_name):
            block = self._conv2d(input, out_dim, kernel_size, kernel_size, strides_size, strides_size)
            block = self._batch_norm(block, train=is_train, name='bn')
            if self.activation == 'sigmoid':
                block = tf.nn.sigmoid(block)
            elif self.activation == 'leaky_relu':
                block = tf.nn.leaky_relu(block)
            elif self.activation == 'tanh':
                block = tf.nn.tanh(block)
            elif self.activation == 'relu':
                block = tf.nn.relu6(block, 'relu6')
            return block

    def _pwise_block(self, input, output_dim, is_train, block_name, bias=False):
        with tf.variable_scope(block_name):
            out = self._conv_1x1(input, output_dim, bias=bias, name='pwb')
            out = self._batch_norm(out, train=is_train, name='bn')
            if self.activation == 'sigmoid':
                block = tf.nn.sigmoid(out)
            elif self.activation == 'leaky_relu':
                block = tf.nn.leaky_relu(out)
            elif self.activation == 'tanh':
                block = tf.nn.tanh(out)
            elif self.activation == 'relu':
                block = tf.nn.relu6(out, 'relu6')
            return block


    def _conv2d(self, input, output_dim, kernel_height, kernel_width, strides_h, strides_w, stddev=0.02, bias=False, name='conv2d'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [kernel_height, kernel_width, input.get_shape()[-1], output_dim],
                regularizer = tc.layers.l2_regularizer(weight_decay),
                initializer = tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input, w, strides=[1, strides_h, strides_w, 1], padding='SAME')
            if bias:
                biases = tf.get_variable('bias', [output_dim], initializer = tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
            return conv

    def _dwise_conv(self, input, kernel_height=3, kernel_width=3, channel_multiplier=1, strides=[1,1,1,1], padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
        with tf.variable_scope(name):
            in_channel = input.get_shape().as_list()[-1]
            w = tf.get_variable('w', [kernel_height, kernel_width, in_channel, channel_multiplier],
                        regularizer=tc.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)
            if bias:
                biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, biases)
            return conv

    def _batch_norm(self, x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
        return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, scale=True, training=train, name=name)

    def _conv_1x1(self, input, output_dim, name, bias=False):
        with tf.name_scope(name):
            return self._conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

    def _global_avg(self, x):
        with tf.name_scope('global_avg'):
            net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
            return net

    def train(self, logits, labels):
        labels_paddings = labels[0:self.batch_size,:]
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_paddings, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope(self.optimzier+'_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == 'Adam':
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.optimzier == 'SGD':
                    self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.optimzier == 'Adagrad':
                    self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(cross_entropy_cost)
                elif self.optimzier == 'Momentum':
                    self.train_op = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize(cross_entropy_cost)
        return self.train_op

    def evaluate(self, logits, labels):
        with tf.name_scope('accuracy_'+self.net_name):
            pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            self.eval_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.eval_op

    def isCompleteTrain(self):
        if (self.cur_epoch == self.desire_epochs) and (self.cur_step == self.desire_steps):
            return True
        else:
            return False

    def getModelInstance(self):
        return self.net_name

    def getModelLogit(self):
        return self.model_logit

    def getCurStep(self):
        return self.cur_step

    def getCurEpoch(self):
        return self.cur_epoch

    def getDesireSteps(self):
        return self.desire_steps

    def getDesireEpochs(self):
        return self.desire_epochs

    def getTrainOp(self):
        return self.train_op

    def getEvalOp(self):
        return self.eval_op

    def setCurStep(self, cur_step=1):
        self.cur_step += cur_step
        if self.cur_step > self.desire_steps:
            self.cur_step = 0
            self.cur_epoch += 1 

    def setCurEpoch(self, cur_epoch=1):
        self.cur_epoch += cur_epoch    

    def setDesireSteps(self, desire_steps):
        self.desire_steps = desire_steps

    def setDesireEpochs(self, desire_epochs):
        self.desire_epochs = desire_epochs

    def setBatchSize(self, batch_size):
        return self.batch_size.assign(batch_size)

    def resetCurStep(self):
        self.cur_step = 0

    def resetCurEpoch(self):
        self.cur_epoch = 0

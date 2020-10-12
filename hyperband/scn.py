import tensorflow as tf

# simple convolutional neural network
class scn(object):
    def __init__(self, net_name, model_layer, input_h, input_w, channel_num, num_classes, batch_size, opt, learning_rate=0.0001, activation='relu'):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.num_classes = num_classes
        self.batch_size = tf.Variable(batch_size)
        self.optimzier = opt
        self.activation = activation
        self.learn_rate = learning_rate
        self.input_size = input_h * input_w * channel_num
        self.cur_step = 1
        self.cur_epoch = 1
        self.desire_steps = -1
        self.desire_epochs = -1
        self.train_op = None
        self.eval_op = None
        self.model_logit = None
    
    def weight_variable(self, shape):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable('W',dtype=tf.float32, shape=shape, initializer=initer)

    def bias_variable(self, shape):   
        initial = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.get_variable('b', dtype=tf.float32, initializer=initial)

    def conv_layer(self, x, filter_size, num_filters, stride, name):
        with tf.variable_scope(name):
            num_in_channel = x.get_shape().as_list()[-1]
            shape = [filter_size, filter_size, num_in_channel, num_filters]
            W = self.weight_variable(shape=shape)
            b = self.bias_variable(shape=[num_filters])
            layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
            layer += b
            return tf.nn.relu(layer)

    def max_pool(self, x, ksize, stride, name):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME", name=name)

    def flatten_layer(self, layer):
        with tf.variable_scope('flatten_layer'):
            layer_shape = layer.get_shape()
            num_features = layer_shape[1:4].num_elements()
            layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat

    def fc_layer(self, x, num_units, name, use_activation=True):
        with tf.variable_scope(name):
            in_dim = x.get_shape()[1]
            W = self.weight_variable(shape=[in_dim, num_units])
            b = self.bias_variable(shape=[num_units])
            layer = tf.matmul(x, W)
            layer += b
            if use_activation:
                if self.activation == 'sigmoid':
                    layer = tf.nn.sigmoid(layer)
                elif self.activation == 'leaky_relu':
                    layer = tf.nn.leaky_relu(layer)
                elif self.activation == 'tanh':
                    layer = tf.nn.tanh(layer)
                elif self.activation == 'relu':
                    layer = tf.nn.relu(layer)
        return layer

    def build(self, input):
        input_padding = input[0:self.batch_size,:,:,:]
        with tf.variable_scope(self.net_name + '_instance'):
            conv1 = self.conv_layer(input_padding, filter_size=5, num_filters=16, stride=1, name='conv1')
            pool1 = self.max_pool(conv1, ksize=2, stride=2, name='pool1')
            if self.model_layer_num >= 1:
                for midx in range(self.model_layer_num - 1):
                    conv1 = self.conv_layer(pool1, filter_size=5, num_filters=16, stride=1, name=str(midx)+'_conv')
                    pool1 = self.max_pool(conv1, ksize=2, stride=2, name=str(midx)+'_pool')
            
            conv2 = self.conv_layer(pool1, filter_size=5, num_filters=32, stride=1, name='conv2')
            pool2 = self.max_pool(conv2, ksize=2, stride=2, name='pool2')
            layer_flat = self.flatten_layer(pool2)
            fc1 = self.fc_layer(layer_flat, num_units=128, name='fc1', use_activation=True)
            self.model_logit = self.fc_layer(fc1, num_units=self.num_classes, name='logit', use_activation=False)
        return self.model_logit

    def train(self, logits, labels):
        labels_paddings = labels[0:self.batch_size,:]
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_paddings, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope(self.optimzier+'_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == 'Adam':
                    self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(cross_entropy_cost)
                elif self.optimzier == 'SGD':
                    self.train_op = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(cross_entropy_cost)
                elif self.optimzier == 'Adagrad':
                    self.train_op = tf.train.AdagradOptimizer(self.learn_rate).minimize(cross_entropy_cost)
                elif self.optimzier == 'Momentum':
                    self.train_op = tf.train.MomentumOptimizer(self.learn_rate,0.9).minimize(cross_entropy_cost)

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
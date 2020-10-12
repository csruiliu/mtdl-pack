import tensorflow as tf

class mlp(object):
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

    def perceptron_layer(self, input):
        weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        layer = tf.matmul(input, weights) + biases
        return layer 

    def build(self, input):
        input_padding = input[0:self.batch_size,:,:,:]
        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input_padding, [-1, self.input_size])
            layer = self.perceptron_layer(input_image)
            if self.model_layer_num >= 1:
                for _ in range(self.model_layer_num - 1):
                    layer = self.perceptron_layer(input = layer)

            if self.activation == 'sigmoid':
                self.model_logit = tf.nn.sigmoid(layer)
            elif self.activation == 'leaky_relu':
                self.model_logit = tf.nn.leaky_relu(layer)
            elif self.activation == 'tanh':
                self.model_logit = tf.nn.tanh(layer)
            elif self.activation == 'relu':
                self.model_logit = tf.nn.relu(layer)

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

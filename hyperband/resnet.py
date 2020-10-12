import tensorflow as tf

class resnet(object):
    def __init__(self, net_name, model_layer, input_h, input_w, channel_num, num_classes, batch_size, opt, learning_rate=0.0001, activation='relu'):
        self.net_name = net_name
        self.model_layer_num = model_layer
        self.img_h = input_h
        self.img_w = input_w
        self.num_classes = num_classes
        self.batch_size = tf.Variable(batch_size)
        self.optimzier = opt
        self.learning_rate = learning_rate
        self.activation = activation
        self.cur_step = 1
        self.cur_epoch = 1
        self.desire_steps = -1
        self.desire_epochs = -1
        self.train_op = None
        self.eval_op = None
        self.model_logit = None

    def conv_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride):
        block_name = 'resnet' + stage + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            x_shortcut = X_input
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            if self.activation == 'sigmoid':
                X = tf.nn.sigmoid(X)
            elif self.activation == 'leaky_relu':
                X = tf.nn.leaky_relu(X)
            elif self.activation == 'tanh':
                X = tf.nn.tanh(X)
            elif self.activation == 'relu':
                X = tf.nn.relu(X)
            

            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            if self.activation == 'sigmoid':
                X = tf.nn.sigmoid(X)
            elif self.activation == 'leaky_relu':
                X = tf.nn.leaky_relu(X)
            elif self.activation == 'tanh':
                X = tf.nn.tanh(X)
            elif self.activation == 'relu':
                X = tf.nn.relu(X)

            W_conv3 = self.weight_variable([1,1,f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            add = tf.add(x_shortcut, X)
            if self.activation == 'sigmoid':
                add_result = tf.nn.sigmoid(add)
            elif self.activation == 'leaky_relu':
                add_result = tf.nn.leaky_relu(add)
            elif self.activation == 'tanh':
                add_result = tf.nn.tanh(add)
            elif self.activation == 'relu':
                add_result = tf.nn.relu(add)
            
        return add_result

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        block_name = 'resnet_' + stage + '_' + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            if self.activation == 'sigmoid':
                X = tf.nn.sigmoid(X)
            elif self.activation == 'leaky_relu':
                X = tf.nn.leaky_relu(X)
            elif self.activation == 'tanh':
                X = tf.nn.tanh(X)
            elif self.activation == 'relu':
                X = tf.nn.relu(X)

            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            if self.activation == 'sigmoid':
                X = tf.nn.sigmoid(X)
            elif self.activation == 'leaky_relu':
                X = tf.nn.leaky_relu(X)
            elif self.activation == 'tanh':
                X = tf.nn.tanh(X)
            elif self.activation == 'relu':
                X = tf.nn.relu(X)

            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            add = tf.add(X, X_shortcut)
            if self.activation == 'sigmoid':
                add_result = tf.nn.sigmoid(add)
            elif self.activation == 'leaky_relu':
                add_result = tf.nn.leaky_relu(add)
            elif self.activation == 'tanh':
                add_result = tf.nn.tanh(add)
            elif self.activation == 'relu':
                add_result = tf.nn.relu(add)

        return add_result

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def build(self, input, training=True, keep_prob=0.5):
        with tf.variable_scope(self.net_name + '_instance'):
            input_padding = input[0:self.batch_size,:,:,:]
            x = tf.pad(input_padding, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
            #training = tf.placeholder(tf.bool, name='training')
            w_conv1 = self.weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            if self.activation == 'sigmoid':
                x = tf.nn.sigmoid(x)
            elif self.activation == 'leaky_relu':
                x = tf.nn.leaky_relu(x)
            elif self.activation == 'tanh':
                x = tf.nn.tanh(x)
            elif self.activation == 'relu':
                x = tf.nn.relu(x)
            
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            #stage 64
            x = self.conv_block(x, 3, 64, [64, 64, 256], stage='stage64', block='conv_block', training=training, stride=1)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage='stage64', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage='stage64', block='identity_block2', training=training)

            #stage 128
            x = self.conv_block(x, 3, 256, [128, 128, 512], stage='stage128', block='conv_block', training=training, stride=2)
            x = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block2', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], stage='stage128', block='identity_block3', training=training)

            #stage 256
            x = self.conv_block(x, 3, 512, [256, 256, 1024], stage='stage256', block='conv_block', training=training, stride=2)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block2', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block3', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block4', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], stage='stage256', block='identity_block5', training=training)

            #stage 512
            x = self.conv_block(x, 3, 1024, [512, 512, 2048], stage='stage512', block='conv_block', training=training, stride=2)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], stage='stage512', block='identity_block1', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], stage='stage512', block='identity_block2', training=training)
            
            avg_pool_size = int(self.img_h // 32)

            x = tf.nn.avg_pool(x, [1, avg_pool_size, avg_pool_size, 1], strides=[1,1,1,1], padding='VALID')

            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            
            with tf.name_scope('dropout'):
                #keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)

            self.model_logit = tf.layers.dense(x, units=self.num_classes, activation=tf.nn.softmax)
            
        return self.model_logit
    
    def train(self, logits, labels):
        labels_padding = labels[0:self.batch_size,:]
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)

        with tf.name_scope(self.optimzier+'_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
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


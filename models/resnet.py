import tensorflow as tf

#resnet-50
class resnet(object):
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

    def weight_variable(self, w_name, shape):
        init_w = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable(name=w_name, dtype=tf.float32, shape=shape, initializer=init_w)

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        f1, f2, f3 = out_filters
        with tf.variable_scope('resnet_' + stage + '_' + block):
            X_shortcut = X_input

            W_conv1 = self.weight_variable('w_conv1', [1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            self.num_conv_layer += 1
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv2 = self.weight_variable('w_conv2', [kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            self.num_conv_layer += 1
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv3 = self.weight_variable('w_conv3', [1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            self.num_conv_layer += 1
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result


    def conv_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride):
        f1, f2, f3 = out_filters
        with tf.variable_scope('resnet' + stage + '_' + block):
            x_shortcut = X_input

            W_conv1 = self.weight_variable('w_conv1',[1, 1, in_filter, f1])
            
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
            self.num_conv_layer += 1
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv2 = self.weight_variable('w_conv2',[kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            self.num_conv_layer += 1
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            W_conv3 = self.weight_variable('w_conv3', [1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            self.num_conv_layer += 1
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            W_shortcut = self.weight_variable('w_shortcut',[1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')
            self.num_conv_layer += 1

            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result


    def build(self, input):
        training = True
        keep_prob = 0.5
        if self.batch_padding:
            input = input[0:self.batch_size, :, :, :]

        with tf.variable_scope(self.net_name + '_instance'):
            x = tf.pad(input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
            w_conv1 = self.weight_variable('w_conv1', [7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.num_pool_layer += 1

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

            x = tf.nn.avg_pool(x, [1, avg_pool_size, avg_pool_size, 1], strides=[1, 1, 1, 1], padding='VALID')
            self.num_pool_layer += 1

            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            self.num_dense_layer += 1

            with tf.name_scope('dropout'):
                x = tf.nn.dropout(x, keep_prob)

            self.model_logit = tf.layers.dense(x, units=self.num_classes, activation=tf.nn.softmax)
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

        with tf.name_scope(self.opt + '_' + self.net_name):
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

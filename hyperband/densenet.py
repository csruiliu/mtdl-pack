import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.keras.layers import GlobalAveragePooling2D

#DenseNet-121
class densenet(object):
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
        self.growth_k = 32
        self.cur_step = 1
        self.cur_epoch = 1
        self.desire_steps = -1
        self.desire_epochs = -1
        self.train_op = None
        self.eval_op = None
        self.model_logit = None

    def bottleneck_block(self, input, block_name):
        with tf.variable_scope(block_name):
            block = tf.layers.batch_normalization(input, training=True, trainable=True, name=block_name+'_bn_0')
            if self.activation == 'sigmoid':
                block = tf.nn.sigmoid(block, name=block_name+'_sigmoid_0')
            elif self.activation == 'leaky_relu':
                block = tf.nn.leaky_relu(block, name=block_name+'_leakyrelu_0')
            elif self.activation == 'tanh':
                block = tf.nn.tanh(block, name=block_name+'_tanh_0')
            elif self.activation == 'relu':
                block = tf.nn.relu(block, name=block_name+'_relu_0')
            
            block = tf.layers.conv2d(block, filters=2*self.growth_k, kernel_size=(1,1), strides=1, padding='same', name=block_name+'_conv_0')

            block = tf.layers.batch_normalization(block, training=True, trainable=True, name=block_name+'_bn_1')
            if self.activation == 'sigmoid':
                block = tf.nn.sigmoid(block, name=block_name+'_sigmoid_1')
            elif self.activation == 'leaky_relu':
                block = tf.nn.leaky_relu(block, name=block_name+'_leakyrelu_1')
            elif self.activation == 'tanh':
                block = tf.nn.tanh(block, name=block_name+'_tanh_1')
            elif self.activation == 'relu':
                block = tf.nn.relu(block, name=block_name+'_relu_1')
            block = tf.layers.conv2d(block, filters=2*self.growth_k, kernel_size=(3,3), strides=1, padding='same', name=block_name+'_conv_1')

            return block

    def dense_block(self, input, dn_layers, block_name):
        with tf.variable_scope(block_name): 
            block = self.bottleneck_block(input, block_name=block_name+'_bottleneck_block_0')
            block_input = tf.concat(values=[input, block],axis=3)
            for i in range(dn_layers - 1):
                block = self.bottleneck_block(block_input, block_name=block_name+'_bottleneck_block_'+str(i+1))
                block_input = tf.concat([block_input,block],axis=3)
            
            return block
    
    def transition_block(self, input, block_name):
        with tf.variable_scope(block_name):
            block = tf.layers.batch_normalization(input, training=True, trainable=True, name=block_name+'_bn_0')
            block = tf.layers.conv2d(block, filters=2*self.growth_k, kernel_size=(1,1), strides=1, padding='same', name=block_name+'_conv_0')
            block = tf.layers.average_pooling2d(block, pool_size=(2,2), strides=2, padding='same', name=block_name+'_avgpool_0')
        
            return block

    def build(self, input):
        instance_name = self.net_name + '_instance'
        with tf.variable_scope(instance_name):
            input_padding = input[0:self.batch_size,:,:,:]
            net = tf.layers.conv2d(input_padding, filters=2*self.growth_k, kernel_size=(7,7), strides=2, padding='same', name='conv_0')
            net = tf.layers.max_pooling2d(net, pool_size=(3,3), strides=2, padding='same', name='max_pool_0')
            net = self.dense_block(net, dn_layers=6, block_name='dense_block_0')
            net = self.transition_block(net, block_name='trans_block_0')
            net = self.dense_block(net, dn_layers=12, block_name='dense_block_1')
            net = self.transition_block(net, block_name='trans_block_1')
            net = self.dense_block(net, dn_layers=24, block_name='dense_block_2')
            net = self.transition_block(net, block_name='trans_block_2')
            net = self.dense_block(net, dn_layers=16, block_name='dense_block_3')
            net = GlobalAveragePooling2D()(net)
            net = tc.layers.flatten(net)
            self.model_logit = tf.layers.dense(net, units=self.num_classes, name='full_connected')
            return self.model_logit
    
    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            labels_padding = labels[0:self.batch_size,:]
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels_padding, logits=logits)
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

    
    

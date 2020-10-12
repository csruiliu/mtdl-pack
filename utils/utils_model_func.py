import tensorflow as tf

#####################################
# activation function
#####################################


def activation_function(logit, act_name):
    new_logit = None
    if act_name == 'relu':
        new_logit = tf.nn.relu(logit, 'relu')
    elif act_name == 'leaky_relu':
        new_logit = tf.nn.leaky_relu(logit, alpha=0.2, name='leaky_relu')
    elif act_name == 'tanh':
        new_logit = tf.math.tanh(logit, 'tanh')
    elif act_name == 'sigmoid':
        new_logit = tf.math.sigmoid(logit, 'sigmoid')
    elif act_name == 'elu':
        new_logit = tf.nn.elu(logit, 'elu')
    elif act_name == 'selu':
        new_logit = tf.nn.selu(logit, 'selu')

    return new_logit

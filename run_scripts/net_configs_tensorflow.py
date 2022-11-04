import keras
import keras.models as models
import numpy as np
import tensorflow as tf
    
    
def ge4j_ge4t_ttH_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)
    
    w1 = tf.compat.v1.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.compat.v1.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.compat.v1.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.compat.v1.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.compat.v1.get_variable('w3', initializer=weights['DenseLayer_2_1/kernel:0'])
    b3 = tf.compat.v1.get_variable('b3', initializer=weights['DenseLayer_2_1/bias:0'])
    w4 = tf.compat.v1.get_variable('w4', initializer=weights['DenseLayer_3_1/kernel:0'])
    b4 = tf.compat.v1.get_variable('b4', initializer=weights['DenseLayer_3_1/bias:0'])
    w5 = tf.compat.v1.get_variable('w5', initializer=weights['outputLayer_1/kernel:0'])
    b5 = tf.compat.v1.get_variable('b5', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.leaky_relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.leaky_relu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.leaky_relu(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.leaky_relu(tf.add(b4, tf.matmul(l3, w4)))

    f = tf.nn.softmax(tf.add(b5, tf.matmul(l4, w5)))
    return f
    
def ge4j_ge3t_ttH_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)
    
    w1 = tf.compat.v1.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.compat.v1.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.compat.v1.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.compat.v1.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.compat.v1.get_variable('w3', initializer=weights['DenseLayer_2_1/kernel:0'])
    b3 = tf.compat.v1.get_variable('b3', initializer=weights['DenseLayer_2_1/bias:0'])
    w4 = tf.compat.v1.get_variable('w4', initializer=weights['DenseLayer_3_1/kernel:0'])
    b4 = tf.compat.v1.get_variable('b4', initializer=weights['DenseLayer_3_1/bias:0'])
    w5 = tf.compat.v1.get_variable('w5', initializer=weights['outputLayer_1/kernel:0'])
    b5 = tf.compat.v1.get_variable('b5', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.leaky_relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.leaky_relu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.leaky_relu(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.leaky_relu(tf.add(b4, tf.matmul(l3, w4)))

    f = tf.nn.softmax(tf.add(b5, tf.matmul(l4, w5)))
    return f



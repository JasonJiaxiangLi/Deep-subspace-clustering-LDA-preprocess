#! /usr/share/Anaconda3/bin python
'''
Created on 2017/9/11
@author: sallysyw & Jasonljx
'''
import tensorflow as tf
import numpy as np
import sklearn.cluster as cluster
# import SCIPY_GET_C
import img2matrix
import tf_get_cs
import prediction_result

# import YaleB
k = 10  # number of the groups
image_per_person = 25  # images per group
yaleb = img2matrix.batch_convert_YaleB(truncate_num=k, images_per_person=image_per_person)
# yaleb = img2matrix.batch_convert_YaleB(truncate_num=1)

train_image = yaleb[0][0]
train_label = yaleb[0][1]
test_image = yaleb[1][0]
test_label = yaleb[1][1]
img_size = yaleb[2]

kmeans_cluster = cluster.KMeans(n_clusters=k).fit(train_image)
result_label = [label + 1 for label in kmeans_cluster.labels_]
true_prediction = np.sum(train_label == result_label)

learning_rate = 0.5     # learning rate
training_epochs = 200000   # training epochs
display_step = 1        # display steps
gamma = 1
lambda1 = 0.1
lambda2 = 0.01
n_input = img_size[0]     # number of inputs
t = int(n_input / 2)   # dimension of LDA compressed data
n_output = t

l=3
n_hidden_1 = int(n_input / np.power(2, 1.0/l))  # number of neurons on the first hidden layer
n_hidden_2 = int(n_input / np.power(2, 2.0/l))  # second hidden layer
n_hidden_3 = t   # third hidden layer

print(n_input,n_hidden_1,n_hidden_2,n_hidden_3)

# tf Graph input
X = tf.Variable(train_image, name="train_data", dtype=tf.float32)

# initialize weights and biases
weights = {
    'encoder_h1': tf.get_variable(name='encoder_h1', shape=[n_input, n_hidden_1], initializer=tf.random_normal_initializer),
    'encoder_h2': tf.get_variable(name='encoder_h2', shape=[n_hidden_1, n_hidden_2], initializer=tf.random_normal_initializer),
    'encoder_h3': tf.get_variable(name='encoder_h3', shape=[n_hidden_2, n_hidden_3], initializer=tf.random_normal_initializer),
    'decoder_h1': tf.get_variable(name='decoder_h1', shape=[n_hidden_3, n_hidden_2], initializer=tf.random_normal_initializer),
    'decoder_h2': tf.get_variable(name='decoder_h2', shape=[n_hidden_2, n_hidden_1], initializer=tf.random_normal_initializer),
    'decoder_h3': tf.get_variable(name='decoder_h3', shape=[n_hidden_1, n_output], initializer=tf.random_normal_initializer),
}

biases = {
    'encoder_b1': tf.get_variable(name='encoder_b1', shape=[n_hidden_1], initializer=tf.random_normal_initializer),
    'encoder_b2': tf.get_variable(name='encoder_b2', shape=[n_hidden_2], initializer=tf.random_normal_initializer),
    'encoder_b3': tf.get_variable(name='encoder_b3', shape=[n_hidden_3], initializer=tf.random_normal_initializer),
    'decoder_b1': tf.get_variable(name='decoder_b1', shape=[n_hidden_2], initializer=tf.random_normal_initializer),
    'decoder_b2': tf.get_variable(name='decoder_b2', shape=[n_hidden_1], initializer=tf.random_normal_initializer),
    'decoder_b3': tf.get_variable(name='decoder_b3', shape=[n_output], initializer=tf.random_normal_initializer),
}


# encoding
def encoder(x): 
    # activation function: sigmoid
    # layer = x*weights['encoder_h1']+biases['encoder_b1']
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3


# decoding
def decoder(x):
    # activation function: sigmoid
    # layer = x*weights['decoder_h1']+biases['decoder_b1']
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3

# the model inference
encoder_op = encoder(X)
encoder_result = encoder_op
decoder_op = decoder(encoder_op)


get = tf_get_cs.tf_getcs(train_image, train_label, t, gamma)
C_0, S = get.get_cs()

# output prediction
y_pred = decoder_op
# actual image
y_true = tf.matmul(X, S)

# define cost and optimizer
# cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
cost1 = 0.5 * tf.norm(y_true - y_pred, 2)/n_input
cost2 = 0.5 * lambda1 * tf.norm(encoder_op - tf.matmul(C_0,encoder_op))/n_input
cost3 = 0.5 * lambda2 * tf.reduce_mean([tf.norm(tf.abs(weights[weight]),2) for weight in weights]+[tf.norm(tf.abs(biases[bias]),1) for bias in biases])
cost=cost1+cost2+cost3

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialization
init = tf.global_variables_initializer()

# run Graph
with tf.Session() as sess:
    sess.run(init)
    # get = SCIPY_GET_C.getcs(train_image, train_label, t, gamma)
    # print(res)
    C=tf.convert_to_tensor(C_0)
    print("Now derive subspace by NN, using C and S")
    for epoch in range(training_epochs):
        batch_xs = train_image
        # C = tf.convert_to_tensor(SCIPY_GET_C.GET_C(np.transpose(batch_xs), batch_size, gamma))
        _, c = sess.run([optimizer, cost])
        # print(sess.run([cost1, cost2, cost3],feed_dict={X:batch_xs}))
        if epoch % display_step == 0:
            print("Epoch:", '%06d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))   
    print("Optimization Finished!")
    # H is the result for clustering
    H = sess.run(encoder_result)

    kmeans_cluster = cluster.KMeans(n_clusters=k).fit(H)
    result_label = [label + 1 for label in kmeans_cluster.labels_]
    print("finally, we have the cluster labels:\n", result_label)
    print("\nwhile the true labels are:\n", train_label)
    true_pred = prediction_result.true_prediction(result_label, k, image_per_person)
    print("\nthe number of true prediction is ", true_pred)


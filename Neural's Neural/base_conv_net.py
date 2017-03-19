# Import tensorflow
import tensorflow as tf
# Import numpy

from dataset_generation import graph_input
from  dataset_generation import labels
# Import input data from csv files
#data = genfromtxt('dataset.csv',delimiter =',' )

# Parameters
learning_rate = 0.01
training_iters = 500
#batch_size = 2
#display_step = 2

# Network Parameters
no_input = 1024 # matrix dataset input (adj_mat shape: max 32*32)
no_classes = 2  # output classes -(dense/sparse)
#dropout = 0.75 # Dropout, probability to keep units

# tf computation graph input
x = tf.placeholder(tf.float32, [no_input])   # input matrices
y = tf.placeholder(tf.float32, [no_classes]) # labels
#keep_prob = tf.placeholder(tf.float32)             #dropout (keep probability)



#  wrappers for convolution and pooling
'''we are using a single stride for convolution without any padding'''
def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

'''reducing the size with pooling using k=2'''
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Weights and Biases dictionaries
'''
#first conv layer ->
->5*5 pixel stride for convolution
->input in sigle matrix
->output of first conv layer wil be 32 features

# seconf conv layer ->
->5*5 pixel stride for convolution
-> input is 32 features from first conv layer
->output will be 64 features

#fully-connected layer ->
-> input is (8*8 result matrix of 2nd pooling )* 64 features
-> output to 1024 neurons

#output layer -> gives classification of input matrix
-> input is from 1024 neurons of fully-connected layer
-> output will be no of classes ie. 2 (dense/sparse)
'''
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 8*8*64 inputs, 1024 outputs
    'wfc1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, no_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bfc1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([no_classes]))
}

# Create conv_net model
def conv_net(x, weights, biases):
    # Reshape input matrix
    x = tf.reshape(x, shape=[-1, 32, 32, 1])

    # first convolution Layer

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # second convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wfc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)

    # Apply dropout to fit the output of fc1 to output layer
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    #print("computed output is",out)
    return out

'''
Now we call our conv_net () function
->parameters 1. input matrix batch in placeholder x
             2. weights dictionary
             3. biases dictionary
             4. dropout probability
'''

# Construct the model by calling the conv_net function
predicted_output = conv_net(x, weights, biases)

# Define loss and optimizer
'''
calculating the cost using cross_entropy function
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predicted_output, tf.cast(y, tf.float32)))
#print('cross-entropy-->',tf.nn.softmax_cross_entropy_with_logits(predicted_output, tf.cast(y, tf.float32)) )
#print("cost is",cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.cast(tf.argmax(predicted_output, 1),tf.float32), tf.cast(y, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the computation graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    #Training until reach max iterations
    x_list = []
    y_list = []
    while step  < training_iters:
        del x_list[:]
        del y_list[:]
        x_list = graph_input()
        y_list = labels()
        sess.run(optimizer, feed_dict={x: x_list, y: y_list})


        # calculating  loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: x_list,
                                                           y: y_list})
        if step % 10 == 0:
            print("iteration",step)
            print(", current Loss= " + \
                  "{:.5f}".format(loss) + ", training accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization done with %d samples !"% (training_iters))

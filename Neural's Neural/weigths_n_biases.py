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
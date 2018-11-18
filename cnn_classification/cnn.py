import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN class for sentence classification
    With an embedding layer + a convolutional, max-pooling and softmax layer
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """

        :param sequence_length: The length of our sentences
        :param num_classes:     Number of classes in the output layer(pos and neg)
        :param vocab_size:      The size of our vocabulary
        :param embedding_size:  The dimensionality of our embeddings.
        :param filter_sizes:    The number of words we want our convolutional filters to cover
        :param num_filters:     The number of filters per filter size
        :param l2_reg_lambda:   optional
        """
        # set placeholders for variables
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='weight')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor
            # with dimensions corresponding to batch, width, height and channel.
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # conv + max-pooling for each filter
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # conv layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1,1,1,1],
                                    padding='VALID', name='conv')
                # activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # max pooling
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length-filter_size + 1, 1, 1],
                                        strides=[1,1,1,1], padding='VALID', name='pool')
                pooled_outputs.append(pooled)


        # combine all the pooled fratures
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)  # why 3?
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #score and prediction
        with tf.name_scope("output"):
            W = tf.get_variable('W', shape=[num_filters_total, num_classes],
                                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.score = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.prediction = tf.argmax(self.score, 1, name='prediction')

        # mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')





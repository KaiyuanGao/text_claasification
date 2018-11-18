import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from BaseUtil.BaseModel import BaseModel


class RNN(BaseModel):
    """
    A RNN class for sentence classification
    With an embedding layer + Bi-LSTM layer + FC layer + softmax
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embed_size, learning_rate, decay_steps, decay_rate,
                 hidden_size, is_training, l2_lambda, grad_clip,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """
        :param sequence_length:
        :param num_classes:
        :param vocab_size:
        :param embedding_size:
        :param learning_rate:
        :param decay_steps:
        :param decay_rate:
        :param hidden_size:
        :param is_training:
        :param l2_lambda:
        :param grad_clip:
        :param initializer:
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.l2_lambda = l2_lambda
        self.grad_clip = grad_clip
        self.initializer = initializer

        # define placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.epoch_step = tf.Variable(0, name='epoch_step', trainable=False)
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.instantiate_weight()
        self.logits = self.inference()

        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

        correct_prediction = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')


    def instantiate_weight(self):
        """define all the weights"""
        with tf.name_scope('weights'):
            self.Embedding = tf.get_variable('Embedding',shape=[self.vocab_size,self.embed_size],
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable('W_projection', shape=[self.hidden_size * 2, self.num_classes],
                                                    initializer=self.initializer)
            self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes])


    def inference(self):
        """
        1. embedding layer
		2. Bi-LSTM layer
		3. concat Bi-LSTM output
		4. FC(full connected) layer
		5. softmax layer
        """
        # embedding layer
        with tf.name_scope('embedding'):
            self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # Bi-LSTM layer
        with tf.name_scope('Bi-LSTM'):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)

            if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                    self.embedded_words,
                                                                    dtype=tf.float32)
            output = tf.concat(outputs, axis=2)
            output_last = tf.reduce_mean(output, axis=1)

            # FC layer
            with tf.name_scope('output'):
                self.score = tf.matmul(output_last, self.W_projection) + self.b_projection
            return self.score

    def loss(self):
        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.score)
            data_loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(cand_v) for cand_v in tf.trainable_variables()
                                    if 'bias' not in cand_v.name]) * self.l2_lambda
            data_loss += l2_loss
            return data_loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.decay_steps, self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss_val)

        #for idx, (grad, var) in enumerate(grads_and_vars):
            #if grad is not None:
                #grads_and_vars[idx] = (tf.clip_by_global_norm(grad, self.grad_clip), var)
        grads_and_vars = [(tf.clip_by_norm(grad, self.grad_clip), val) for grad, val in grads_and_vars]
        #grads_and_vars = [(tf.add(grad, tf.random_normal(tf.shape(grad), stddev=self.config.grad_noise)), val) for grad, val in
               #gvs]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op


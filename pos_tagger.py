import tensorflow as tf
#import numpy as np

class PoSTagger(object):
    """
    A simple PoS tagger implementation in Tensorflow.
    Uses an embedding layer followed by a fully connected layer with ReLU and a softmax layer.
    """
    def __init__(self, n_pos_tags, vocab_size, embedding_size, n_past_words): # sequence_length, filter_sizes, num_filters, l2_reg_lambda=0.0

        print("Initialising PoSTagger...")
        print("n_pos_tags: ", n_pos_tags)

        # Minibatch placeholders for input and output
        # The word indices of the window
        self.input_x = tf.placeholder(tf.int32, [None, n_past_words+1], name="input_x")
        # The target pos-tags
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y") 

        with tf.device('/gpu:0'):
            
            # Embedding layer
            with tf.name_scope("embedding"):
                # Create an embedding matrix
                # Initialise following recommendations from
                # https://www.tensorflow.org/get_started/mnist/pros
                self.embedding_matrix = tf.Variable(
                    tf.truncated_normal(
                        [vocab_size, embedding_size],
                        stddev=0.1
                    )
                )
 
            # Fully connected layer with ReLU 
            with tf.name_scope("model"):

                # Create feature vector

                self.word_matrix = \
                    tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
                # stack the rows
                # -1: account for variable batch size
                # TODO: understand
                new_shape = [-1, (n_past_words + 1) * embedding_size]
                self.feature_vector = tf.reshape(self.word_matrix, new_shape)

                # send feature vector through hidden layers

                feature_vector_size = (n_past_words + 1) * embedding_size
                h1_size = 100
                w1 = tf.Variable(
                    tf.truncated_normal(
                        [feature_vector_size, h1_size],
                        stddev=0.1
                    )
                )
                self.h1 = tf.nn.relu(
                    tf.matmul(self.feature_vector, w1)
                )

                """
                h2_size = 100
                self.w2 = tf.Variable(
                    tf.truncated_normal(
                        [h1_size, h2_size],
                        stddev=0.1
                    )
                )
                self.h2 = tf.nn.relu(
                    tf.matmul(self.h1, self.w2)
                )
                """

                # Compute softmax logits 

                self.wl = tf.Variable(
                    tf.truncated_normal(
                        [h1_size, n_pos_tags],
                        stddev=0.1
                    )
                )
                self.logits = tf.matmul(self.h1, self.wl)
    
                # Compute the mean loss using tf.nn.sparse_softmax_cross_entropy_with_logits

                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.input_y,
                        logits=self.logits
                    )
                )

            # Calculate accuracy
            with tf.name_scope("accuracy"):
                # compute the average accuracy over the batch (remember tf.argmax and tf.equal)

                # logits has shape [?, 42]
                self.predictions = tf.argmax(self.logits, axis=1)
                correct_prediction = tf.equal(self.predictions, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                    tf.float32))

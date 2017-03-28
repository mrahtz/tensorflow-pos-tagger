import tensorflow as tf
#import numpy as np

class PoSTagger(object):
    """
    A simple PoS tagger implementation in Tensorflow.
    Uses an embedding layer followed by a fully connected layer with ReLU and a softmax layer.
    """
    def __init__(self, n_pos_tags, vocab_size, embedding_size, n_past_words): # sequence_length, filter_sizes, num_filters, l2_reg_lambda=0.0
        # Minibatch placeholders for input and output
        # The word indices of the window
        self.input_x = tf.placeholder(tf.int32, [None, n_past_words+1], name="input_x")
        # The target pos-tags
        self.input_y = tf.placeholder(tf.int64, [None    ], name="input_y") 
            
        with tf.device('/gpu:0'):
            
            # Embedding layer
            with tf.name_scope("embedding"):
                # TODO Create an embedding matrix
                # first, create a one-hot matrix with '1's corresponding
                # to the current list of words
                # e.g. for input_x = [0, 2],
                # create a matrix
                # [1, 0, 0;
                #  0, 0, 1] 
                # depth is the width of the matrix;
                # the height is determined by the number of indices
                # so this matrix is len(input_x) x depth
                #                 = n_past_words x embedding_size
                one_hot = tf.one_hot(indices=input_x, depth=embedding_size)

                embedding_matrix = \
                    tf.Variable(tf.zeros(vocab_size, embedding_size))

                # so this guy will be (n_past_words + 1) x embedding size
                word_matrix = one_hot * embedding_matrix
                # now we just need to stack the rows of this guy
                embedding_concatenation = np.concat(word_matrix)
                
            # Fully connected layer with ReLU 
            with tf.name_scope("model"):
                # Create feature vector
                feature_vector = embedding_concatenation

                # send feature vector through hidden layer
                feature_vector_size = (n_past_words + 1) * embedding_size
                hidden_layer_size = 100
                weight_matrix = \
                        tf.Variable(
                            tf.zeros(hidden_layer_size, feature_vector_size)
                        )

                hidden_layer = tf.nn.relu(weight_matrix * feature_vector)

                # Compute softmax logits 
                w_m_2 = tf.variable(tf.zeros(n_pos_tags, hidden_layer_size))
                self.logits = w_m_2 * hidden_layer
                
    
                # Compute the mean loss using tf.nn.sparse_softmax_cross_entropy_with_logits
                # result is a 1D tensor of length batch_size
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    _sentinel=None,
                    labels=self.input_y,
                    logits=self.logits)

            # Calculate accuracy
            with tf.name_scope("accuracy"):
                # compute the average accuracy over the batch (remember tf.argmax and tf.equal)
                # TODO what is this for?
                #self.predictions = 
                self.accuracy = tf.reduce_mean(self.loss)

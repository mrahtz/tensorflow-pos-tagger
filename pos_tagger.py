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
        self.input_y = tf.placeholder(tf.int64, [None    ], name="input_y") 

        print("input_x has shape", self.input_x.get_shape())
            
        with tf.device('/gpu:0'):
            
            # Embedding layer
            with tf.name_scope("embedding"):
                # Create an embedding matrix

                """
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
                one_hot = tf.one_hot(indices=self.input_x, depth=vocab_size)
                """

                embedding_matrix = \
                    tf.Variable(tf.zeros([vocab_size, embedding_size]))
                print("embedding_matrix has shape",
                        embedding_matrix.get_shape())

                """
                # so this guy will be (n_past_words + 1) x embedding size
                word_matrix = one_hot * embedding_matrix
                """
                word_matrix = \
                    tf.nn.embedding_lookup(embedding_matrix, self.input_x)
                print("word_matrix has shape", word_matrix.get_shape())

                # now we just need to stack the rows of this guy
                # -1: account for variable batch size
                # TODO: understand
                new_shape = [-1, (n_past_words + 1) * embedding_size]
                embedding_concatenation = tf.reshape(word_matrix, new_shape)
                print("embedding_concatenation has shape",
                        embedding_concatenation.get_shape())
                
            # Fully connected layer with ReLU 
            with tf.name_scope("model"):
                # Create feature vector
                feature_vector = embedding_concatenation
                print("feature_vector has shape", feature_vector.get_shape())  

                # send feature vector through hidden layer
                feature_vector_size = (n_past_words + 1) * embedding_size
                hidden_layer_size = 100
                # TODO: what's the right shape here?
                weight_matrix = \
                        tf.Variable(
                            tf.zeros([feature_vector_size, hidden_layer_size])
                        )
                print("weight_matrix has shape", weight_matrix.get_shape())

                mult = tf.matmul(feature_vector, weight_matrix)
                print("mult has shape", mult.get_shape())

                hidden_layer = tf.nn.relu(mult)
                print("hidden_layer has shape", hidden_layer.get_shape())

                # Compute softmax logits 
                w_m_2 = tf.Variable(tf.zeros([hidden_layer_size, n_pos_tags]))
                print("w_m_2 has shape", w_m_2.get_shape())
                self.logits = tf.matmul(hidden_layer, w_m_2)
                print("logits has shape", self.logits.get_shape())
    
                # Compute the mean loss using tf.nn.sparse_softmax_cross_entropy_with_logits
                # result is a 1D tensor of length batch_size
                # NB sparse_softmax: allows us to specify correct class
                # as an index
                self.loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.input_y,
                            logits=self.logits
                ))

            # Calculate accuracy
            with tf.name_scope("accuracy"):
                # compute the average accuracy over the batch (remember tf.argmax and tf.equal)

                # logits has shape [?, 42]
                self.predictions = tf.argmax(self.logits, axis=1)
                correct_prediction = tf.equal(self.predictions, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                    tf.float32))

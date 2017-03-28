import tensorflow as tf
#import numpy as np

class PoSTagger(object):
	"""
	A simple PoS tagger implementation in Tensorflow.
	Uses an embedding layer followed by a fully connected layer with ReLU and a softmax layer.
	"""
	def __init__(self, num_classes, vocab_size, embedding_size, past_words): # sequence_length, filter_sizes, num_filters, l2_reg_lambda=0.0
		# Minibatch placeholders for input and output
		# The word indices of the window
		self.input_x = tf.placeholder(tf.int32, [None, past_words+1], name="input_x")
		# The target pos-tags
		self.input_y = tf.placeholder(tf.int64, [None	], name="input_y") 
			
		with tf.device('/gpu:0'):
			
		
			
	
			# Embedding layer
			with tf.name_scope("embedding"):
				# TODO Create an embedding matrix
				
			# Fully connected layer with ReLU 
			with tf.name_scope("model"):
				# TODO Create feature vector
				
				# TODO send feature vector through hidden layer
				
				# TODO Compute softmax logits 
				self.logits = ...
	
				# TODO Compute the mean loss using tf.nn.sparse_softmax_cross_entropy_with_logits
				self.loss = 
	
			# Calculate accuracy
			with tf.name_scope("accuracy"):
				# TODO compute the average accuracy over the batch (remember tf.argmax and tf.equal)
				self.predictions = ....
				self.accuracy = ...

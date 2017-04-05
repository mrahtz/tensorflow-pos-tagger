import data_utils
from pos_tagger import PoSTagger

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle

from data_utils import TextLoader

sentence = input('Enter a sentence to be annotated: ')
words = sentence.strip().split(" ")
textloader = TextLoader()
features = textloader.parse(words)

sess = tf.Session()

checkpoint_file = tf.train.latest_checkpoint('runs/1491414625/checkpoints/')
saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
saver.restore(sess, checkpoint_file)

graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

predicted_pos_ids = sess.run(predictions, feed_dict={input_x: features})
predicted_pos = textloader.pos_ids_to_pos(predicted_pos_ids)

word_pos_tuples = zip(words, predicted_pos)
annotated_words = []
for tup in word_pos_tuples:
    annotated_word = '%s/%s' % (tup[0], tup[1])
    annotated_words.append(annotated_word)
annotated_sentence = ' '.join(annotated_words)
print("Your sentence, annotated:")
print(annotated_sentence)

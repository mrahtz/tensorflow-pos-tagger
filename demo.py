import data_utils
from pos_tagger import PoSTagger

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle


sentence = input('Enter a sentence to be annotated: ')
textloader = data_utils.TextLoader(sentence, vocab_path='/tmp/vocab.pkl', vocab_size=50000, n_past_words=3)

sess = tf.Session()

checkpoint_file = tf.train.latest_checkpoint('runs/1491424895/checkpoints/')
saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
saver.restore(sess, checkpoint_file)

graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

predicted_pos_ids = \
    sess.run(predictions, feed_dict={input_x: textloader.features})

words = []
for sentence_word_ids in textloader.features:
    word_id = sentence_word_ids[0]
    words.append(textloader.id_to_word[word_id])
predicted_pos = []
for pred_id in predicted_pos_ids:
    predicted_pos.append(textloader.id_to_pos[pred_id])

word_pos_tuples = zip(words, predicted_pos)
annotated_words = []
for tup in word_pos_tuples:
    annotated_word = '%s/%s' % (tup[0], tup[1])
    annotated_words.append(annotated_word)
annotated_sentence = ' '.join(annotated_words)
print("Your sentence, annotated:")
print(annotated_sentence)

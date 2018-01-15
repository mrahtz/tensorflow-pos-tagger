# tensorflow-pos-tagger
# Copyright (C) 2017 Matthew Rahtz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import data_utils

import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import pickle

CACHE_DIR = 'cache'
vocab_path = os.path.join(CACHE_DIR, 'vocab.pkl')

if not os.path.exists(vocab_path):
    print("Error: vocabulary file '%s' doesn't exist." % vocab_path)
    print("Train the model first using train.py.")
    sys.exit(1)

sentence = input('Enter a sentence to be annotated:\n')
print()
textloader = data_utils.TextLoader(
    sentence, vocab_size=50000, n_past_words=3, vocab_path=vocab_path
)

sess = tf.Session()

checkpoint_file = tf.train.latest_checkpoint('checkpoints/')
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

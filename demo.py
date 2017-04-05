import data_utils
from pos_tagger import PoSTagger

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle

from data_utils import TextLoader

words = "I like fish because I am a cat"
words = "David Ivor St Hubbins is a fictional character"
textloader = TextLoader()
features = textloader.parse(words)

## EVALUATION ##

graph = tf.Graph()
with graph.as_default():

    session_conf = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        checkpoint_file = tf.train.latest_checkpoint('runs/1491410780/checkpoints/')
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

        predicted_pos_ids = sess.run(predictions, {input_x: features})
        predicted_pos = textloader.pos_ids_to_pos(predicted_pos_ids)
        print(predicted_pos)

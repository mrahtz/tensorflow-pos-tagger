import data_utils
from pos_tagger import PoSTagger

import tensorflow as tf
import numpy as np
import os
import time
import datetime

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "/data/corpus-01", "Path to the test data")
# Model parameters
tf.flags.DEFINE_integer("past_words", 3, "How many previous words are used for prediction (default: 3)")
# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1490130308/checkpoints/", "Checkpoint directory from training run")
# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

## DATA PREPARATION ##

# Load data
print("Loading and preprocessing test dataset \n")
x_test, y_test = data_utils.load_data_and_labels_test(FLAGS.data_file_path, FLAGS.past_words)

## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("accuracy").outputs[0]

        # Generate batches for one epoch
        batches = data_utils.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
correct_predictions = float(sum(all_predictions == y_test))
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

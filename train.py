import data_utils
import model

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import numpy as np

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")

tf.flags.DEFINE_string("data_file_path", "data/corpus.small", "Path to the training data")
# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of word embeddings (default: 128)")
tf.flags.DEFINE_integer("vocab_size", 50000, "Size of the vocabulary (default: 50k)")
tf.flags.DEFINE_integer("past_words", 3, "How many previous words are used for prediction (default: 3)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
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

with open(FLAGS.data_file_path, 'r') as f:
    tagged_sentences = f.read()
textloader = data_utils.TextLoader(tagged_sentences, 'vocab/vocab.pkl',
        FLAGS.vocab_size, FLAGS.past_words, tensor_path='vocab/tensors.pkl')

x = textloader.features
y = textloader.labels
num_outputTags = len(textloader.pos_to_id)


idx = int(FLAGS.dev_sample_percentage * len(x))
x_test, x_train = x[:idx], x[idx:]
y_test, y_train = y[:idx], y[idx:]

# Generate training batches
batches = data_utils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
print("Done \n")
## MODEL AND TRAINING PROCEDURE DEFINITION ##

sess = tf.Session()

# num_outputTags: the number of unique POS tags seen in the
# training data

# Initialize model
pos_tagger = model.Tagger(
    n_pos_tags=num_outputTags, 
    vocab_size=FLAGS.vocab_size, 
    embedding_size=FLAGS.embedding_dim, 
    n_past_words=FLAGS.past_words
)

# Define training procedure

# first argument: the value of the variable
global_step = tf.Variable(0, name="global_step", trainable=False)
# Define an optimizer, e.g. AdamOptimizer
optimizer = tf.train.AdamOptimizer()
# Define an optimizer step
train_op = optimizer.minimize(pos_tagger.loss, global_step=global_step)

timestamp = int(time.time())
log_dir = 'logs/%d/' % timestamp
os.makedirs(log_dir)

checkpoint_dir = 'checkpoints/'

# Add ops to record summaries for loss and accuracy...
train_loss = tf.summary.scalar("train_loss", pos_tagger.loss)
train_accuracy = tf.summary.scalar("train_accuracy", pos_tagger.accuracy)
# ...then merge these ops into one single op so that they easily be run together
train_summary_ops = tf.summary.merge([train_loss, train_accuracy])
# Same ops, but with different names, so that train/test results show up
# separately in TensorBoard
test_loss = tf.summary.scalar("test_loss", pos_tagger.loss)
test_accuracy = tf.summary.scalar("test_accuracy", pos_tagger.accuracy)
test_summary_ops = tf.summary.merge([test_loss, test_accuracy])

# (this step also writes the graph to the events file so that
# it shows up in TensorBoard)
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

# Initialize all variables
sess.run(tf.global_variables_initializer())
sess.graph.finalize()

def step(x, y, train):
    feed_dict = {
        pos_tagger.input_x: x,
        pos_tagger.input_y: y
    }
    ops = [global_step, pos_tagger.loss, pos_tagger.accuracy]

    if train:
        _, summaries, step, loss, accuracy = sess.run(
            [train_op, train_summary_ops] + ops, feed_dict
        )
    else:
        summaries, step, loss, accuracy = sess.run(
            [test_summary_ops] + ops, feed_dict
        )

    print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy))
    summary_writer.add_summary(summaries, step)

for batch in batches:
    x_batch, y_batch = zip(*batch)
    step(x_batch, y_batch, train=True)
    current_step = tf.train.global_step(sess, global_step)

    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        step(x_test, y_test, train=False)
        print("")

    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to '%s'" % path)

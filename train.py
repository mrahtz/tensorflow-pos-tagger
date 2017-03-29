import data_utils
from pos_tagger import PoSTagger

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import numpy as np

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")

tf.flags.DEFINE_string("data_file_path", "data/corpus.128", "Path to the training data")
# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embeddings (default: 128)")
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

print("Loading and preprocessing traning and dev datasets \n")
if os.path.isfile('cache.pkl'):
    with open('cache.pkl', 'rb') as f:
        x, y, num_outputTags = pickle.load(f)
else:
    # Load data
    x, y, num_outputTags = data_utils.load_data_and_labels(FLAGS.data_file_path, FLAGS.vocab_size, FLAGS.past_words)
    with open('cache.pkl', 'wb') as f:
        pickle.dump((x, y, num_outputTags), f)

# Randomly shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(len(y))
x_shuffled = x[shuffled_indices]
y_shuffled = y[shuffled_indices]

# Split train/dev sets
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# Generate training batches
batches = data_utils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
print("Done \n")
## MODEL AND TRAINING PROCEDURE DEFINITION ##

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        # allow_soft_placement: allow TensorFlow to deploy itself on whatever it
        # thinks is sensible, if the requested device isn't available
        # (pos_tagger.py requests GPU)
        allow_soft_placement=FLAGS.allow_soft_placement,
        # log_device_placement: make a note in the log of which device
        # TensorFlow depoloyed itself on (which CPU/GPU)
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # num_outputTags: the number of unique POS tags seen in the
        # training data

        # Initialize model
        pos_tagger = PoSTagger(
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

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", pos_tagger.loss)
        acc_summary = tf.summary.scalar("accuracy", pos_tagger.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        # Define training and dev steps (batch) 
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                pos_tagger.input_x: x_batch,
                pos_tagger.input_y: y_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, pos_tagger.loss, pos_tagger.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)



        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                pos_tagger.input_x: x_batch,
                pos_tagger.input_y: y_batch
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, pos_tagger.loss, pos_tagger.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        ## TRAINING LOOP ##
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

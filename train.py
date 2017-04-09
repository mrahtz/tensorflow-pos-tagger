import data_utils
import model

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import numpy as np

CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")

tf.flags.DEFINE_string("data_file_path", "data/corpus.supersmall", "Path to the training data")
# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of word embeddings (default: 128)")
tf.flags.DEFINE_integer("vocab_size", 50000, "Size of the vocabulary (default: 50k)")
tf.flags.DEFINE_integer("n_past_words", 3, "How many previous words are used for prediction (default: 3)")
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


def load_data():
    with open(FLAGS.data_file_path, 'r') as f:
        tagged_sentences = f.read()
    textloader = data_utils.TextLoader(
        tagged_sentences,
        FLAGS.vocab_size, FLAGS.n_past_words,
        vocab_path='vocab/vocab.pkl', tensor_path='vocab/tensors.pkl'
    )

    x = textloader.features
    y = textloader.labels
    n_pos_tags = len(textloader.pos_to_id)

    idx = int(FLAGS.dev_sample_percentage * len(x))
    x_test, x_train = x[:idx], x[idx:]
    y_test, y_train = y[:idx], y[idx:]

    train_batches = data_utils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    test_data = {'x': x_test, 'y': y_test}

    return (train_batches, test_data, n_pos_tags)


def model_init(vocab_size, embedding_size, n_past_words, n_pos_tags):
    pos_tagger = model.Tagger(
        vocab_size, embedding_size, n_past_words, n_pos_tags
    )

    global_step = tf.Variable(
        initial_value=0, name="global_step", trainable=False
    )
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(pos_tagger.loss, global_step=global_step)

    return pos_tagger, train_op, global_step


def logging_init(model, graph):
    """
    Set up logging so that progress can be visualised in TensorBoard.
    """
    # Add ops to record summaries for loss and accuracy...
    train_loss = tf.summary.scalar("train_loss", model.loss)
    train_accuracy = tf.summary.scalar("train_accuracy", model.accuracy)
    # ...then merge these ops into one single op so that they easily be run
    # together
    train_summary_ops = tf.summary.merge([train_loss, train_accuracy])
    # Same ops, but with different names, so that train/test results show up
    # separately in TensorBoard
    test_loss = tf.summary.scalar("test_loss", model.loss)
    test_accuracy = tf.summary.scalar("test_accuracy", model.accuracy)
    test_summary_ops = tf.summary.merge([test_loss, test_accuracy])

    timestamp = int(time.time())
    run_log_dir = os.path.join(LOGS_DIR, str(timestamp))
    os.makedirs(run_log_dir)
    # (this step also writes the graph to the events file so that
    # it shows up in TensorBoard)
    summary_writer = tf.summary.FileWriter(run_log_dir, graph)

    return train_summary_ops, test_summary_ops, summary_writer


def checkpointing_init():
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    return saver


def step(sess, model,
        standard_ops, train_ops, test_ops,
        x, y,
        summary_writer,
        train):
    feed_dict = {
        model.input_x: x,
        model.input_y: y
    }

    if train:
        step, loss, accuracy, _, summaries = sess.run(
            standard_ops + train_ops, feed_dict
        )
    else:
        step, loss, accuracy, summaries = sess.run(
            standard_ops + test_ops, feed_dict
        )

    print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy))
    summary_writer.add_summary(summaries, step)


def main():
    sess = tf.Session()

    train_batches, test_data, n_pos_tags = load_data()
    x_test = test_data['x']
    y_test = test_data['y']
    pos_tagger, train_op, global_step = model_init(
            FLAGS.vocab_size, FLAGS.embedding_dim,
            FLAGS.n_past_words, n_pos_tags
    )
    train_summary_ops, test_summary_ops, summary_writer = logging_init(
        pos_tagger, sess.graph
    )
    saver = checkpointing_init()

    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    standard_ops = [global_step, pos_tagger.loss, pos_tagger.accuracy]
    train_ops = [train_op, train_summary_ops]
    test_ops = [test_summary_ops]

    for batch in train_batches:
        x_batch, y_batch = zip(*batch)
        step(
            sess,
            pos_tagger, standard_ops, train_ops, test_ops,
            x_batch, y_batch,
            summary_writer,
            train=True,
        )
        current_step = tf.train.global_step(sess, global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            step(
                sess,
                pos_tagger, standard_ops, train_ops, test_ops,
                x_test, y_test,
                summary_writer,
                train=False
            )
            print("")

        if current_step % FLAGS.checkpoint_every == 0:
            prefix = os.path.join(CHECKPOINT_DIR, 'model')
            path = saver.save(sess, prefix, global_step=current_step)
            print("Saved model checkpoint to '%s'" % path)

main()

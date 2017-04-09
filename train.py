import os
import time
import argparse

import tensorflow as tf

import data_utils
import model

CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'
CACHE_DIR = 'cache'
# Evaluate model on training set every this number of steps
EVALUATE_EVERY = 100
# Save a checkpoint every this number of steps
CHECKPOINT_EVERY = 100


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_proportion",
        type=float,
        default=0.1,
        help="Proportion of the training data to reserve for validation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/corpus.small",
        help="Path to the training data")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=50,
        help="Dimensionality of word embeddings")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Number of words to remember in vocabulary")
    parser.add_argument(
        "--n_past_words",
        type=int,
        default=3,
        help="Number of previous words to use for prediction")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="Number of training epochs")

    args = parser.parse_args()
    return args


def load_data(data_path, vocab_size, n_past_words, test_proportion, batch_size,
              n_epochs):
    with open(data_path, 'r') as f:
        tagged_sentences = f.read()

    vocab_path = os.path.join(CACHE_DIR, 'vocab.pkl')
    tensor_path = os.path.join(CACHE_DIR, 'tensors.pkl')

    textloader = data_utils.TextLoader(
        tagged_sentences,
        vocab_size,
        n_past_words,
        vocab_path,
        tensor_path)

    x = textloader.features
    y = textloader.labels
    n_pos_tags = len(textloader.pos_to_id)

    idx = int(test_proportion * len(x))
    x_test, x_train = x[:idx], x[idx:]
    y_test, y_train = y[:idx], y[idx:]

    train_batches = data_utils.batch_iter(
        list(zip(x_train, y_train)), batch_size, n_epochs)
    test_data = {'x': x_test, 'y': y_test}

    return (train_batches, test_data, n_pos_tags)


def model_init(vocab_size, embedding_size, n_past_words, n_pos_tags):
    pos_tagger = model.Tagger(vocab_size, embedding_size, n_past_words,
                              n_pos_tags)

    global_step = tf.Variable(
        initial_value=0, name="global_step", trainable=False)
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
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    return saver


def step(sess, model, standard_ops, train_ops, test_ops, x, y, summary_writer,
         train):
    feed_dict = {model.input_x: x, model.input_y: y}

    if train:
        step, loss, accuracy, _, summaries = sess.run(standard_ops + train_ops,
                                                      feed_dict)
    else:
        step, loss, accuracy, summaries = sess.run(standard_ops + test_ops,
                                                   feed_dict)

    print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy))
    summary_writer.add_summary(summaries, step)


def main():
    args = parse_args()

    sess = tf.Session()

    train_batches, test_data, n_pos_tags = load_data(
        args.data_path, args.vocab_size, args.n_past_words,
        args.test_proportion, args.batch_size, args.n_epochs)
    x_test = test_data['x']
    y_test = test_data['y']
    pos_tagger, train_op, global_step = model_init(
        args.vocab_size, args.embedding_dim, args.n_past_words, n_pos_tags)
    train_summary_ops, test_summary_ops, summary_writer = logging_init(
        pos_tagger, sess.graph)
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
            pos_tagger,
            standard_ops,
            train_ops,
            test_ops,
            x_batch,
            y_batch,
            summary_writer,
            train=True)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % EVALUATE_EVERY == 0:
            print("\nEvaluation:")
            step(
                sess,
                pos_tagger,
                standard_ops,
                train_ops,
                test_ops,
                x_test,
                y_test,
                summary_writer,
                train=False)
            print("")

        if current_step % CHECKPOINT_EVERY == 0:
            prefix = os.path.join(CHECKPOINT_DIR, 'model')
            path = saver.save(sess, prefix, global_step=current_step)
            print("Saved model checkpoint to '%s'" % path)


main()

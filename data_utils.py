import numpy as np
import os
from collections import Counter
import pickle

UNKNOWN_WORD_ID = 0
UNKNOWN_WORD_SYMBOL = "<UNKNOWN WORD>"

class TextLoader():

    def __init__(self, tagged_sentences, vocab_path, vocab_size, n_past_words):
        self.vocab_size = vocab_size
        self.n_past_words = n_past_words

        if os.path.exists(vocab_path):
            print("Loading saved vocabulary...")
            self.load_vocab(vocab_path)
        else:
            print("Generating vocabulary...")
            self.gen_vocab(tagged_sentences)
            self.save_vocab(vocab_path)

        self.features, self.labels = \
            self.get_features_and_labels(tagged_sentences)


    def gen_vocab(self, tagged_sentences):
        words, pos_tags = self.split_sentence(tagged_sentences)

        word_counts = Counter(words)
        unique_pos_tags = set(pos_tags)

        # most_common() returns (word, count) tuples
        words_to_keep = [t[0] for t in word_counts.most_common(self.vocab_size)]

        self.word_to_id = \
            {word: i for i, word in enumerate(words_to_keep, start=1)}
        # add unknown token to vocabulary
        # (all words not contained in it will be mapped to this)
        self.word_to_id[UNKNOWN_WORD_SYMBOL] = UNKNOWN_WORD_ID

        self.pos_to_id = \
            {pos: i for i, pos in enumerate(list(unique_pos_tags))}

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_pos = {v: k for k, v in self.pos_to_id.items()}

        self.words = words


    def save_vocab(self, vocab_filename):
        dicts = [self.word_to_id,
                self.pos_to_id,
                self.id_to_word,
                self.id_to_pos]
        with open(vocab_filename, 'wb') as f:
            pickle.dump(dicts, f)


    def load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            dicts = pickle.load(f)
        self.word_to_id = dicts[0]
        self.pos_to_id = dicts[1]
        self.id_to_word = dicts[2]
        self.id_to_pos = dicts[3]


    def get_features_and_labels(self, tagged_sentences):
        x = []
        y = []

        for sentence in tagged_sentences.split('\n'):
            words, pos_tags = self.split_sentence(sentence)

            for j in range(len(words)):
                if len(pos_tags) != 0:
                    tag = pos_tags[j]
                    y.append(self.pos_to_id[tag])

                past_word_ids = []
                for k in range(0, self.n_past_words+1):
                    if j-k < 0: # out of bounds
                        past_word_ids.append(UNKNOWN_WORD_ID)
                    elif words[j-k] in self.word_to_id:
                        past_word_ids.append(self.word_to_id[words[j-k]])
                    else: # word not in vocabulary
                        past_word_ids.append(UNKNOWN_WORD_ID)
                x.append(past_word_ids)

        return x, y


    def split_sentence(self, tagged_sentence):
        tagged_words = tagged_sentence.split()
        word_tag_tuples = [x.split("/") for x in tagged_words]

        words = [t[0] for t in word_tag_tuples]
        # If we have an unannotated sentence, we'll only have the words,
        # so the length of the tuple will be just 1
        pos_tags = [t[1] for t in word_tag_tuples if len(t) == 2]

        # len(pos_tags) == 0: unannotated sentence
        # But if we have an annotated sentence, we should have the same
        # number of words and tags
        if len(pos_tags) != 0 and len(pos_tags) != len(words):
            raise ValueError("Number of words doesn't match number of tags")

        return words, pos_tags

    def pos_ids_to_pos(self, pos_ids):
        pos = []
        for pos_id in pos_ids:
            pos.append(self.id_to_pos_dict[pos_id])
        return pos


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


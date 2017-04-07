import numpy as np
import os
from collections import Counter
import pickle

UNKNOWN_WORD_ID = 0
UNKNOWN_WORD = "<UNKNOWN_WORD>"

UNTAGGED_POS_ID = 0
UNTAGGED_POS = "<UNTAGGED_POS>"

class TextLoader():

    def __init__(self, sentences, vocab_path, vocab_size, n_past_words):
        self.vocab_size = vocab_size
        self.n_past_words = n_past_words

        if os.path.exists(vocab_path):
            print("Loading saved vocabulary...")
            self.load_vocab(vocab_path)
        else:
            print("Generating vocabulary...")
            self.gen_vocab(tagged_sentences)
            self.save_vocab(vocab_path)

        print("Generating tensors...")
        self.features, self.labels = \
            self.get_features_and_labels(sentences)


    def gen_vocab(self, tagged_sentences):
        words, pos_tags = \
            self.split_sentence(tagged_sentences, drop_untagged=True)

        word_counts = Counter(words)
        unique_pos_tags = set(pos_tags)

        # most_common() returns (word, count) tuples
        # Why the '- 1'? To account for the extra word we add for words
        # not in the vocabulary, UNKNOWN_WORD.
        words_to_keep = \
            [t[0] for t in word_counts.most_common(self.vocab_size - 1)]

        self.word_to_id = \
            {word: i for i, word in enumerate(words_to_keep, start=1)}
        # words not in the vocabulary will be mapped to this word
        self.word_to_id[UNKNOWN_WORD] = UNKNOWN_WORD_ID # = 0

        self.pos_to_id = \
            {pos: i for i, pos in enumerate(list(unique_pos_tags), start=1)}
        self.pos_to_id[UNTAGGED_POS] = UNTAGGED_POS_ID # = 0

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
            # Why drop_untagged=False here?
            # Because we might have received an untagged sentence
            # which we now want to tag.
            words, pos_tags = self.split_sentence(sentence, drop_untagged=False)

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


    def split_sentence(self, tagged_sentence, drop_untagged):
        tagged_words = tagged_sentence.split()
        word_tag_tuples = [x.split("/") for x in tagged_words]

        words = []
        pos_tags = []
        for word_tag_tuple in word_tag_tuples:
            if len(word_tag_tuple) > 2:
                # We've got something like AC/DC/NNP
                continue

            if drop_untagged and len(word_tag_tuple) == 1:
                continue

            word = word_tag_tuple[0]
            words.append(word)

            if len(word_tag_tuple) == 1:
                pos_tags.append(UNTAGGED_POS)
            else:
                tag = word_tag_tuple[1]
                pos_tags.append(tag)

        return words, pos_tags


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


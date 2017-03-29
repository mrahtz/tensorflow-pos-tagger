import numpy as np
import os
from collections import Counter
import pickle

def clean_string(string):
    return string.lower()

def load_data_and_labels(data_file_path, max_vocabSize, past_words):
    """
    Loads training data, creates vocabulary and returns the respective ids for words and tags
    
    Returns:
    - x: a list of lists - one list for each word
         each list contains the ID of the word in the vocabulary,
         along with the IDs of the previous words
    - y: the POS tag for each of the words
    """
    # Load data from file
    cwd = os.getcwd()
    # Collect word counts and unique PoS tags
    word_counts = Counter()
    unique_posTags = set()
    with open(data_file_path, "r") as tagged_sentences:
        for sentence in tagged_sentences:
            for tag in sentence.strip().split(" "):
                splitted_tag = tag.split("/")
                if len(splitted_tag) != 2:
                    continue
                word = clean_string(splitted_tag[0])
                pos = splitted_tag[1]
                unique_posTags.add(pos) # collect all unique PoS tags
                if word in word_counts: # collect word frequencies (used later to prune vocabulary)
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    # Prune vocabulary to max_vocabSize
    words_toKeep = [tupl[0] for tupl in word_counts.most_common(max_vocabSize-1)]
    # Create mapping from words/PoS tags to ids
    # (IDs start from 1)
    word_toId = {word: i for i, word in enumerate(words_toKeep, 1)}
    # ID 0: UNK
    word_toId["<UNK>"] = 0 # add unknown token to vocabulary (all words not contained in it will be mapped to this)
    pos_toId = {pos: i for i, pos in enumerate(list(unique_posTags))}
    # Save vocabulary and PoS tags ids for evaluation
    if not os.path.exists(cwd+"/vocab"):
        os.makedirs(cwd+"/vocab")
    with open(cwd+"/vocab/wordIds.pkl", "wb") as f:
        pickle.dump(word_toId, f)
    with open(cwd+"/vocab/posIds.pkl", "wb") as f:
        pickle.dump(pos_toId, f)
    # Replace each word with the IDs of the previous "past_words" words
    # (past_words: int)
    # and replace each PoS tag by its respective id
    x = []
    y = []
    with open(data_file_path, "r") as tagged_sentences:
        for sentence in tagged_sentences:
            pairs = sentence.strip().split(" ")
            words_and_tags = list(pair.split("/") for pair in pairs if len(pair.split("/")) == 2)
            if len(words_and_tags) == 0:
                continue
            words, pos_tags = zip(*words_and_tags)
            words = [clean_string(word) for word in words]
            for j in range(len(words)):
                y.append(pos_toId[ pos_tags[j] ])
                pastWords_ids = []
                for k in range(0, past_words+1): # for previous words
                    if j-k < 0: # out of bounds
                        pastWords_ids.append(0) # <UNK>
                    elif words[j-k] in word_toId: # word in vocabulary
                        pastWords_ids.append(word_toId[ words[j-k] ])
                    else: # word not in vocabulary
                        pastWords_ids.append(0) # <UNK>    
                x.append(pastWords_ids)

    return [np.array(x), np.array(y), len(unique_posTags)]


def load_data_and_labels_test(data_file_path, past_words):
    """
    Loads test data and vocabulary and returns the respective ids for words and tags
    """
    cwd = os.getcwd()

    # Load vocabulary and PoS tags ids from training
    if not os.path.exists(cwd+"/vocab"):
        raise FileNotFoundError("You need to run train.py first in order to generate the vocabulary.")
    with open(cwd+"/vocab/wordIds.pkl", "rb") as f:
        word_toId = pickle.load(f)
    with open(cwd+"/vocab/posIds.pkl", "rb") as f:
        pos_toId = pickle.load(f)
    # Replace each word with the id of the previous "past_words" words
    # and replace each PoS tag by its respective id
    x = []
    y = []
    with open(cwd+data_file_path, "r") as tagged_sentences:
        for sentence in tagged_sentences:
            pairs = sentence.strip().split(" ")
            words, pos_tags = zip(*(pair.split("/") for pair in pairs if len(pair.split("/")) == 2))
            for j in range(len(words)): # for each word in the sentence
                if pos_tags[j] in pos_toId: 
                    y.append(pos_toId[ pos_tags[j] ])
                else:
                    y.append(0) # TODO: This is not correct, but we should have seen all posible output tags in advance...
                pastWords_ids = []
                for k in range(1, past_words+1): # for previous words
                    if j-k < 0: # out of bounds
                        pastWords_ids.append(0) # <UNK>
                    elif words[j-k] in word_toId: # word in vocabulary
                        pastWords_ids.append(word_toId[ words[j-k] ])
                    else: # word not in vocabulary
                        pastWords_ids.append(0) # <UNK>    
                x.append(pastWords_ids)

    return [np.array(x), np.array(y)]


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


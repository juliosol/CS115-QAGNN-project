from urllib.request import urlretrieve
import h5py
import json
import pickle
import os
import numpy as np
import sys

if not os.path.isfile('colab_data/english.h5'):
    print("Downloading Conceptnet Numberbatch word embeddings...")
    conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5'
    urlretrieve(conceptnet_url, 'colab_data/english.h5')



with h5py.File('colab_data/english.h5', 'r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]

print("all_words dimensions: {}".format(len(all_words)))
print("all_embeddings dimensions: {}".format(all_embeddings.shape))
print("Random example word: {}".format(all_words[1337]))

english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
english_embeddings = all_embeddings[english_word_indices]
print("Number of English words in all_words: {0}".format(len(english_words)))
print("english_embeddings dimensions: {0}".format(english_embeddings.shape))
print(english_words[1337])

word_index = {word: i for i, word in enumerate(english_words)}

norms = np.linalg.norm(english_embeddings, axis=1)
normalized_embeddings = english_embeddings.astype('float32') / norms.astype('float32').reshape([-1, 1])

file_name = ['english_words', 'english_word_indices', 'english_embeddings', 'normalized_embeddings', 'word_index']

for name in file_name:

    var = globals()[name]
    with open(f'colab_data/{name}.pickle', 'wb') as pickle_file:
        pickle.dump(var, pickle_file)
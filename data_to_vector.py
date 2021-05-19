import numpy as np
import json
import nltk
from gensim import downloader
from gensim.models import KeyedVectors
import pickle
import utils

##################
## Convert json to location vectors 
## Resave into numpy
#################

word2vec = KeyedVectors.load("word2vec.model")

data_dict = utils.read_data_dict("data/portland_or_500.json")

vec_dict = {}
for k in data_dict.keys():
    v = utils.text_to_vector(word2vec, data_dict[k])
    vec_dict[k] = v

with open("data/portland_or_vec.pkl", "wb") as f:
    pickle.dump(vec_dict, f)
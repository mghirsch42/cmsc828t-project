import numpy as np
import json
import pickle
# import nltk
# from gensim import downloader
# from gensim.models import KeyedVectors

def read_data_dict(data_path):
    with open(data_path, "r") as f:
        data_dict = json.load(f)
    return data_dict

def text_to_vector(word2vec, text):
    sents = text.split("\u2029 ")
    sents = [nltk.word_tokenize(sent) for sent in sents]
    sents = [nltk.pos_tag(sent) for sent in sents]
    locs = []
    for sent in sents:
        ne = nltk.ne_chunk(sent)
        for subtree in ne:
            if hasattr(subtree, "label") and subtree.label() == "GPE":
                locs.append([l[0] for l in subtree.leaves()])
    new_locs = []
    for loc in locs:
        new_loc = " ".join(loc)
        new_locs.append(new_loc)
    locs = new_locs
    loc_vecs = []
    for loc in locs:
        try:
            loc_vecs.append(word2vec[loc])
        except:
            continue
    if len(loc_vecs) == 0:
        avg_vec = np.zeros((300,))
    else:
        avg_vec = np.mean(loc_vecs, axis=0)
    return avg_vec

def data_to_vectors(word2vec, data_dict):
    vecs = []
    for k in data_dict.keys():
        v = text_to_vector(word2vec, data_dict[k])
        vecs.append(v)
    return vecs

def read_data_vector(data_path):
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    vecs = [data_dict[k] for k in data_dict.keys()]
    return vecs

def load_data_vectors():
    texas_vecs = read_data_vector("data/paris_texas_vec.pkl")
    france_vecs = read_data_vector("data/paris_france_vec.pkl")
    maine_vecs = read_data_vector("data/portland_me_vec.pkl")
    oregon_vecs = read_data_vector("data/portland_or_vec.pkl")
    return texas_vecs, france_vecs, maine_vecs, oregon_vecs

def binary_labels(pos, neg1, neg2, neg3):
    return [1]*len(pos), [0]*len(neg1), [0]*len(neg2), [0]*len(neg3)

def multi_class_labels(c0, c1, c2, c3):
    return [0]*len(c0), [1]*len(c1), [2]*len(c2), [3]*len(c3)


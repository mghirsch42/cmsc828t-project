import numpy as np
import json
import nltk
from gensim import downloader
from gensim.models import KeyedVectors


with open("data/paris_texas.json", "r") as f:
    data_dict = json.load(f)

keys = list(data_dict.keys())
text = data_dict[keys[1]]


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
print(new_locs)
locs = new_locs

word2vec = KeyedVectors.load("word2vec.model")


loc_vecs = []
for loc in locs:
    try:
        loc_vecs.append(word2vec[loc])
    except:
        continue
avg_vec = np.mean(loc_vecs, axis=0)

print(loc_vecs)
print(avg_vec)
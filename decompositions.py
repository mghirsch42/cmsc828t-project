import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
import pickle
from matplotlib import pyplot as plt

import utils

#################
## Run decompositions and plot
#################

texas_vecs, france_vecs, maine_vecs, oregon_vecs = utils.load_data_vectors()
texas_labels, france_labels, maine_labels, oregon_labels = utils.multi_class_labels(texas_vecs, france_vecs, maine_vecs, oregon_vecs)

examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))
print(np.shape(examples))
print(np.shape(labels))

pca = PCA(n_components=2)
pca.fit(np.concatenate((texas_vecs, france_vecs)))


t_texas = pca.transform(texas_vecs)
t_france = pca.transform(france_vecs)
# t_maine = pca.transform(maine_vecs)
# t_oregon = pca.transform(oregon_vecs)
plt.plot(t_france[:,0], t_france[:,1], "bo", alpha=.5, label="Paris, France")
# plt.plot(t_oregon[:,0], t_oregon[:,1], "go", alpha=.5, label="Portland, Oregon")
# plt.plot(t_maine[:,0], t_maine[:,1], "yo", alpha=.5, label="Portland, Maine")
plt.plot(t_texas[:,0], t_texas[:,1], "ro", alpha=.5, label="Paris, Texas")
plt.legend()
plt.title("PCA\nParis")
plt.savefig("figures/pca_paris")
plt.show()

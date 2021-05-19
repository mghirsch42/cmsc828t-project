import numpy as np
import pickle
from matplotlib import pyplot as plt
import utils

################
## Calculate word vector statistics, plot
################

plt.rc("font", size=12)
plt.rc("figure", titlesize=16)

texas_vecs, france_vecs, maine_vecs, oregon_vecs = utils.load_data_vectors()

print(list(map(round, texas_vecs[0], [4]*300)))

# texas_avgs = np.mean(texas_vecs, axis=0)
# texas_std = np.std(texas_vecs, axis=0)
# france_avgs = np.mean(france_vecs, axis=0)
# france_std = np.std(france_vecs, axis=0)
# maine_avgs = np.mean(maine_vecs, axis=0)
# maine_std = np.std(maine_vecs, axis=0)
# oregon_avgs = np.mean(oregon_vecs, axis=0)
# oregon_std = np.std(oregon_vecs, axis=0)

# avgs = [texas_avgs, france_avgs, maine_avgs, oregon_avgs]
# std = [texas_std, france_std, maine_std, oregon_std]

# for loc in avgs:
#     print("Averages:")
#     print("Average:", np.mean(france_avgs))
#     print("Std:", np.std(france_avgs))
#     print("---")
#     print("Stds:")
#     print("Average:", np.mean(france_std))
#     print("Std:", np.std(france_std))
#     print("")

# plt.hist(texas_avgs, color="r", alpha=.3, bins=25, label="Paris, Texas")
# plt.hist(france_avgs, color="b", alpha=.3, bins=25, label="Paris, France")
# plt.hist(maine_avgs, color="y", alpha=.3, bins=25, label="Portland, Maine")
# plt.hist(oregon_avgs, color="g", alpha=.3, bins=25, label="Portland, Oregon")
# plt.title("Histogram of Word Vector Values\nParis, Texas")
# plt.title("Histogram of Word Vector Values\nParis, France")
# plt.title("Histogram of Word Vector Values\nPortland, Maine")
# plt.title("Histogram of Word Vector Values\nPortland, Oregon")
# plt.title("Histogram of Word Vector Values\nParis")
# plt.title("Histogram of Word Vector Values\nPortland")
# plt.title("Histogram of Word Vector Values")
# plt.xlabel("Vector Entry Value")
# plt.ylabel("Number of Entries")
# plt.xlim((-.3, .3))
# plt.ylim((0, 40))
# plt.savefig("figures/hist_texas")
# plt.savefig("figures/hist_france")
# plt.savefig("figures/hist_maine")
# plt.savefig("figures/hist_oregon")
# plt.legend()
# plt.savefig("figures/hist_paris")
# plt.savefig("figures/hist_portland")
# plt.savefig("figures/hist_all")
# plt.show()

# plt.figure(figsize=(15, 5))
# plt.bar(np.arange(len(texas_avgs)), texas_avgs, color="r", alpha=.5, label="Paris, Texas")
# plt.bar(np.arange(len(france_avgs)), france_avgs, color="b", alpha=.5, label="Paris, France")
# plt.bar(np.arange(len(maine_avgs)), maine_avgs, color="y", alpha=.75, label="Portland, Maine")
# plt.bar(np.arange(len(oregon_avgs)), oregon_avgs, color="g", alpha=.5, label="Portland, Oregon")
# plt.legend()
# plt.title("Average Word Vector Values\nPortland")
# plt.savefig("figures/word2vec_portland")
# plt.show()


# word2vec = KeyedVectors.load("word2vec.model")

# print(word2vec.most_similar([oregon_avgs], topn=5))
# print(word2vec.most_similar([france_avgs], topn=5))
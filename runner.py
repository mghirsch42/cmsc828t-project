import math

from regression3 import Classifers
from results import Resulter
import utils

texas_vecs, france_vecs, maine_vecs, oregon_vecs = utils.load_data_vectors()
classifier = Classifers(texas_vecs, france_vecs, maine_vecs, oregon_vecs, "results3/")
classes = ["texas", "france", "maine", "oregon"]


# for alg in ["sgd", "lr", "knn", "rf"]:
#     for pos_class in classes:
#         classifier.one_vs_all(alg, pos_class, None)
#         classifier.one_vs_all(alg, pos_class, "balanced")

# for alg in ["lr", "knn", "rf"]:
#     classifier.multi_class(alg, None)
#     classifier.multi_class(alg, "balanced")

# for alg in ["sgd", "lr", "knn", "rf"]:
#     for c1, c2 in [("texas", "france"), ("maine", "oregon")]:
#         classifier.two_class(alg, c1, c2, None)
#         classifier.two_class(alg, c2, c1, None)
#         classifier.two_class(alg, c1, c2, "balanced")
#         classifier.two_class(alg, c2, c1, "balanced")

# resulter = Resulter("results3/")
# for alg in ["sgd", "lr", "knn", "rf"]:
#     resulter.one_vs_all(alg, "balanced")
#     resulter.one_vs_all(alg, None)

# for alg in ["lr", "knn", "rf"]:
#     resulter.multi_class(alg, "balanced")
#     resulter.multi_class(alg, None)

# for alg in ["sgd", "lr", "knn", "rf"]:
#     resulter.two_class_paris(alg, "balanced")
#     resulter.two_class_paris(alg, None)
#     resulter.two_class_portland(alg, "balanced")
#     resulter.two_class_portland(alg, None)


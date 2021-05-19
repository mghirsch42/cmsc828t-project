import numpy as np
import json
import nltk
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import csv
import utils



def sgd(vecs1, vecs2, labels1, labels2, pos_class):
    examples = np.concatenate((vecs1, vecs2))
    labels = np.concatenate((labels1, labels2))
    classifier = SGDClassifier(class_weight="balanced")
    results = []
    for i in range(50):
        classifier = classifier.fit(examples, labels)
        preds = classifier.predict(vecs1)
        res1 = np.sum(preds)/len(preds)
        preds = classifier.predict(vecs2)
        res2 = np.sum(preds)/len(preds)
        results.append([res1, res2])
    with open("results/sgd2_{}_b.csv".format(pos_class), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def bin_lr(vecs1, vecs2, labels1, labels2, pos_class):
    examples = np.concatenate((vecs1, vecs2))
    labels = np.concatenate((labels1, labels2))
    classifier = LogisticRegression(class_weight="balanced")
    results = []
    for i in range(50):
        classifier = classifier.fit(examples, labels)
        preds = classifier.predict(vecs1)
        res1 = np.sum(preds)/len(preds)
        preds = classifier.predict(vecs2)
        res2 = np.sum(preds)/len(preds)
        results.append([res1, res2])
    print(classifier.score(examples, labels))
    print(np.count_nonzero(preds == 1))
    with open("results/lr2_{}_b.csv".format(pos_class), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def bin_knn(vecs1, vecs2, labels1, labels2, pos_class):
    examples = np.concatenate((vecs1, vecs2))
    labels = np.concatenate((labels1, labels2))
    classifier = KNeighborsClassifier()
    results = []
    for i in range(50):
        classifier = classifier.fit(examples, labels)
        preds = classifier.predict(vecs1)
        res1 = np.sum(preds)/len(preds)
        preds = classifier.predict(vecs2)
        res2 = np.sum(preds)/len(preds)
        results.append([res1, res2])
    with open("results/knn2_{}.csv".format(pos_class), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def bin_rf(vecs1, vecs2, labels1, labels2, pos_class):
    examples = np.concatenate((vecs1, vecs2))
    labels = np.concatenate((labels1, labels2))
    results = []
    for i in range(50):
        classifier = RandomForestClassifier(n_estimators=10, max_depth=10, class_weight="balanced")
        classifier = classifier.fit(examples, labels)
        preds = classifier.predict(vecs1)
        res1 = np.sum(preds)/len(preds)
        preds = classifier.predict(vecs2)
        res2 = np.sum(preds)/len(preds)
        results.append([res1, res2])
    with open("results/rf2_{}_b.csv".format(pos_class), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

texas_vecs, france_vecs, maine_vecs, oregon_vecs = utils.load_data_vectors()

# Paris
texas_pos = [1]*len(texas_vecs)
france_neg = [0]*len(france_vecs)
bin_rf(texas_vecs, france_vecs, texas_pos, france_neg, "texas")
texas_neg = [0]*len(texas_vecs)
france_pos = [1]*len(france_vecs)
bin_rf(texas_vecs, france_vecs, texas_neg, france_pos, "france")
# exit()
# Portland
maine_pos = [1]*len(maine_vecs)
oregon_neg = [0]*len(oregon_vecs)
bin_rf(maine_vecs, oregon_vecs, maine_pos, oregon_neg, "maine")
maine_neg = [0]*len(maine_vecs)
oregon_pos = [1]*len(oregon_vecs)
bin_rf(maine_vecs, oregon_vecs, maine_neg, oregon_pos, "oregon")
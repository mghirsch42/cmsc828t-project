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



def sgd(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, pos_class):
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))
    classifier = SGDClassifier(class_weight="balanced")
    results = []
    for i in range(50):
        classifier = classifier.fit(examples, labels)
        score = classifier.score(examples, labels)
        preds = classifier.predict(texas_vecs)
        texas = np.sum(preds)/len(preds)
        preds = classifier.predict(france_vecs)
        france = np.sum(preds)/len(preds)
        preds = classifier.predict(maine_vecs)
        maine = np.sum(preds)/len(preds)
        preds = classifier.predict(oregon_vecs)
        oregon = np.sum(preds)/len(preds)
        results.append([texas, france, maine, oregon])

    # with open("results/sgd_{}_b.csv".format(pos_class), "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(results)

def lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs):
    texas_labels, france_labels, maine_labels, oregon_labels = utils.multi_class_labels(texas_vecs, france_vecs, maine_vecs, oregon_vecs)
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))

    classifier = LogisticRegression(multi_class="multinomial", class_weight="balanced")
    results = []
    for i in range(50):
        classifier.fit(examples, labels)
        preds = classifier.predict(texas_vecs)
        texas = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        # print(texas)
        preds = classifier.predict(france_vecs)
        france = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        # print(france)
        preds = classifier.predict(maine_vecs)
        maine = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        # print(maine)
        preds = classifier.predict(oregon_vecs)
        oregon = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        # print(oregeon)
        results.append([texas, france, maine, oregon])
    print(classifier.score(examples, labels))
    exit()
    with open("results/lr.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def bin_lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, pos_class):
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))
    classifier = LogisticRegression(class_weight="balanced")
    results = []
    for i in range(50):
        classifier = classifier.fit(examples, labels)
        score = classifier.score(examples, labels)
        preds = classifier.predict(texas_vecs)
        texas = np.sum(preds)/len(preds)
        preds = classifier.predict(france_vecs)
        france = np.sum(preds)/len(preds)
        preds = classifier.predict(maine_vecs)
        maine = np.sum(preds)/len(preds)
        preds = classifier.predict(oregon_vecs)
        oregon = np.sum(preds)/len(preds)
        results.append([texas, france, maine, oregon])
        preds = classifier.predict(examples)
    print(classifier.score(examples, labels))
    print(np.count_nonzero(preds == 1))
    with open("results/lr_{}_b.csv".format(pos_class), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs):
    texas_labels, france_labels, maine_labels, oregon_labels = utils.multi_class_labels(texas_vecs, france_vecs, maine_vecs, oregon_vecs)
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))

    results = []
    for i in range(1):
        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(examples, labels)
        print(classifier.score(examples, labels))
        preds = classifier.predict(texas_vecs)
        texas = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        preds = classifier.predict(france_vecs)
        france = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        preds = classifier.predict(maine_vecs)
        maine = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        preds = classifier.predict(oregon_vecs)
        oregon = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        results.append([texas, france, maine, oregon])
        
    # with open("results/knn2.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(results)

def bin_knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, pos_class):
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))
    results = []
    for i in range(1):
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier = classifier.fit(examples, labels)
        score = classifier.score(examples, labels)
        preds = classifier.predict(texas_vecs)
        texas = np.sum(preds)/len(preds)
        preds = classifier.predict(france_vecs)
        france = np.sum(preds)/len(preds)
        preds = classifier.predict(maine_vecs)
        maine = np.sum(preds)/len(preds)
        preds = classifier.predict(oregon_vecs)
        oregon = np.sum(preds)/len(preds)
        results.append([texas, france, maine, oregon])
        preds = classifier.predict(examples)
        print(classifier.score(examples, labels))
        # print(np.count_nonzero(preds == 1))
        # exit()
    # with open("results/knn_{}.csv".format(pos_class), "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(results)

def rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs):
    texas_labels, france_labels, maine_labels, oregon_labels = utils.multi_class_labels(texas_vecs, france_vecs, maine_vecs, oregon_vecs)
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))

    classifier = RandomForestClassifier(n_estimators=10, max_depth=12)
    results = []
    for i in range(50):
        classifier = RandomForestClassifier()
        classifier.fit(examples, labels)
        print(classifier.score(examples, labels))
        preds = classifier.predict(texas_vecs)
        texas = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        preds = classifier.predict(france_vecs)
        france = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        preds = classifier.predict(maine_vecs)
        maine = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        preds = classifier.predict(oregon_vecs)
        oregon = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        results.append([texas, france, maine, oregon])
    with open("results/rf.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def bin_rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, pos_class):
    examples = np.concatenate((texas_vecs, france_vecs, maine_vecs, oregon_vecs))
    labels = np.concatenate((texas_labels, france_labels, maine_labels, oregon_labels))
    results = []
    for i in range(50):
        classifier = RandomForestClassifier(n_estimators=10, max_depth=12)
        classifier = classifier.fit(examples, labels)
        score = classifier.score(examples, labels)
        preds = classifier.predict(texas_vecs)
        texas = np.sum(preds)/len(preds)
        preds = classifier.predict(france_vecs)
        france = np.sum(preds)/len(preds)
        preds = classifier.predict(maine_vecs)
        maine = np.sum(preds)/len(preds)
        preds = classifier.predict(oregon_vecs)
        oregon = np.sum(preds)/len(preds)
        results.append([texas, france, maine, oregon])
        preds = classifier.predict(examples)
        # print(classifier.score(examples, labels))
        # print(np.count_nonzero(preds == 1))
        
    with open("results/rf_{}.csv".format(pos_class), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

texas_vecs, france_vecs, maine_vecs, oregon_vecs = utils.load_data_vectors()
# lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs)
# knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs)
# rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs)

# texas_labels, france_labels, maine_labels, oregon_labels = utils.binary_labels(texas_vecs, france_vecs, maine_vecs, oregon_vecs)
# sgd(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "texas")
# bin_lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "texas")
# bin_knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "texas")
# bin_rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "texas")

# france_labels, texas_labels, maine_labels, oregon_labels = utils.binary_labels(france_vecs, texas_vecs, maine_vecs, oregon_vecs)
# sgd(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "france")
# bin_lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "france")
# bin_knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "france")
# bin_rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "france")

# maine_labels, texas_labels, france_labels, oregon_labels = utils.binary_labels(maine_vecs, texas_vecs, france_vecs, oregon_vecs)
# sgd(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "maine")
# bin_lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "maine")
# bin_knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "maine")
# bin_rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "maine")

oregon_labels, texas_labels, france_labels, maine_labels  = utils.binary_labels(oregon_vecs, texas_vecs, france_vecs, maine_vecs)
# sgd(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "oregon")
# bin_lr(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "oregon")
bin_knn(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "oregon")
# bin_rf(texas_vecs, france_vecs, maine_vecs, oregon_vecs, texas_labels, france_labels, maine_labels, oregon_labels, "oregon")

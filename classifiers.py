import numpy as np
import nltk
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import math

#################
## Methods to run classifiers for different experiments
#################

class Classifers():
    def __init__(self, texas_vecs, france_vecs, maine_vecs, oregon_vecs, result_dir):
        self.texas_vecs = texas_vecs
        self.france_vecs = france_vecs
        self.maine_vecs = maine_vecs
        self.oregon_vecs = oregon_vecs
        self.result_dir = result_dir
        self.split_train_val()
        self.clear_labels()

    def split_train_val(self):        
        texas_split = math.floor(.9 * len(self.texas_vecs))
        self.texas_train = self.texas_vecs[:texas_split]
        self.texas_val = self.texas_vecs[texas_split:]

        france_split = math.floor(.9 * len(self.france_vecs))
        self.france_train = self.france_vecs[:france_split]
        self.france_val = self.france_vecs[france_split:]

        maine_split = math.floor(.9 * len(self.maine_vecs))
        self.maine_train = self.maine_vecs[:maine_split]
        self.maine_val = self.maine_vecs[maine_split:]

        oregon_split = math.floor(.9 * len(self.oregon_vecs))
        self.oregon_train = self.oregon_vecs[:oregon_split]
        self.oregon_val = self.oregon_vecs[oregon_split:]

    def multi_class_labels(self):
        self.texas_labels_train = [0] * len(self.texas_train)
        self.texas_labels_val = [0] * len(self.texas_val)
        self.france_labels_train = [1] * len(self.france_train)
        self.france_labels_val = [1] * len(self.france_val)
        self.maine_labels_train = [2] * len(self.maine_train)
        self.maine_labels_val = [2] * len(self.maine_val)
        self.oregon_labels_train = [3] * len(self.oregon_train)
        self.oregon_labels_val = [3] * len(self.oregon_val)

    def binary_labels(self, pos_class):
        self.clear_labels()      
        if pos_class == "texas":
            self.texas_labels_train = [1] * len(self.texas_train)
            self.texas_labels_val = [1] * len(self.texas_val)
        if pos_class == "france":
            self.france_labels_train = [1] * len(self.france_train)
            self.france_labels_val = [1] * len(self.france_val)
        if pos_class == "maine":
            self.maine_labels_train = [1] * len(self.maine_train)
            self.maine_labels_val = [1] * len(self.maine_val)
        if pos_class == "oregon":
            self.oregon_labels_train = [1] * len(self.oregon_train)
            self.oregon_labels_val = [1] * len(self.oregon_val)

    def clear_labels(self):
        self.texas_labels_train = [0] * len(self.texas_train)
        self.texas_labels_val = [0] * len(self.texas_val)
        self.france_labels_train = [0] * len(self.france_train)
        self.france_labels_val = [0] * len(self.france_val)
        self.maine_labels_train = [0] * len(self.maine_train)
        self.maine_labels_val = [0] * len(self.maine_val)
        self.oregon_labels_train = [0] * len(self.oregon_train)
        self.oregon_labels_val = [0] * len(self.oregon_val)

    def get_examples(self):
        train_examples = np.concatenate((self.texas_train, self.france_train, self.maine_train, self.oregon_train))
        train_labels = np.concatenate((self.texas_labels_train, self.france_labels_train, self.maine_labels_train, self.oregon_labels_train))
        return train_examples, train_labels

    def binary_predict(self, classifier, vecs, pos):
        preds = classifier.predict(vecs)
        res = np.sum(preds)/len(preds)
        if not pos:
            res = 1-res
        return round(res, 3)

    def binary_predict_classes(self, classifier, pos_class):
        # Texas
        texas_res_train = self.binary_predict(classifier, self.texas_train, pos_class=="texas")
        texas_res_val = self.binary_predict(classifier, self.texas_val, pos_class=="texas")
        # France
        france_res_train = self.binary_predict(classifier, self.france_train, pos_class=="france")
        france_res_val = self.binary_predict(classifier, self.france_val, pos_class=="france")
        # Maine
        maine_res_train = self.binary_predict(classifier, self.maine_train, pos_class=="maine")
        maine_res_val = self.binary_predict(classifier, self.maine_val, pos_class=="maine")
        # Oregon
        oregon_res_train = self.binary_predict(classifier, self.oregon_train, pos_class=="oregon")
        oregon_res_val = self.binary_predict(classifier, self.oregon_val, pos_class=="oregon")
        return [texas_res_train, france_res_train, maine_res_train, oregon_res_train], [texas_res_val, france_res_val, maine_res_val, oregon_res_val]

    def mc_predict(self, classifier, vecs):
        preds = classifier.predict(vecs)
        res = [np.count_nonzero(preds==0), np.count_nonzero(preds==1), np.count_nonzero(preds==2), np.count_nonzero(preds==3)]
        return res

    def mc_predict_classes(self, classifier):
        texas_res_train = self.mc_predict(classifier, self.texas_train)
        texas_res_val = self.mc_predict(classifier, self.texas_val)
        france_res_train = self.mc_predict(classifier, self.france_train)
        france_res_val = self.mc_predict(classifier, self.france_val)
        maine_res_train = self.mc_predict(classifier, self.maine_train)
        maine_res_val = self.mc_predict(classifier, self.maine_val)
        oregon_res_train = self.mc_predict(classifier, self.oregon_train)
        oregon_res_val = self.mc_predict(classifier, self.oregon_val)
        return [texas_res_train, france_res_train, maine_res_train, oregon_res_train], [texas_res_val, france_res_val, maine_res_val, oregon_res_val]
 
    def save_results(self, results, fname):
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(results)

    def get_fname(self, alg, pos_class, balanced, res_type=""):
        fname = self.result_dir + alg
        if pos_class:
            fname += "_" + pos_class
        if balanced:
            fname += "_b"
        fname_train = fname + res_type + "_train.csv"
        fname_val = fname + res_type + "_val.csv"
        return fname_train, fname_val

    def one_vs_all(self, alg, pos_class, balanced):
        print("Running ONE-VS-ALL: Alg={}, Pos_class={}, Balanced={}".format(alg, pos_class, balanced))
        self.binary_labels(pos_class)
        train_examples, train_labels = self.get_examples()
        results_train = []
        results_val = []
        results2_train = []
        results2_val = []
        for i in range(50):
            if alg == "sgd": classifier = SGDClassifier(class_weight=balanced)
            elif alg == "lr": classifier = LogisticRegression(class_weight=balanced)
            elif alg == "knn": classifier = KNeighborsClassifier(n_neighbors=3)
            elif alg == "rf": classifier = RandomForestClassifier(n_estimators=10, max_depth=12, class_weight=balanced)
            classifier = classifier.fit(train_examples, train_labels)
            pred_results_train, pred_results_val = self.binary_predict_classes(classifier, pos_class)
            pred_results2_train, pred_results2_val = self.ova_predict(classifier, pos_class)
            results_train.append(pred_results_train)
            results_val.append(pred_results_val)
            results2_train.append(pred_results2_train)
            results2_val.append(pred_results2_val)
        fname_train, fname_val = self.get_fname(alg, pos_class, balanced)
        fname2_train, fname2_val = self.get_fname(alg, pos_class, balanced, "_2")
        print("Saving results to: {}, {}".format(fname_train, fname_val))
        self.save_results(results_train, fname_train)
        self.save_results(results_val, fname_val)
        self.save_results(results2_train, fname2_train)
        self.save_results(results2_val, fname2_val)

    def multi_class(self, alg, balanced):
        print("Running MULTICLASS: Alg={}, Balanced={}".format(alg, balanced))
        self.multi_class_labels()
        examples, labels = self.get_examples()
        results_train = []
        results_val = []
        for i in range(50):
            if alg == "lr": classifier = LogisticRegression(class_weight=balanced)
            elif alg == "knn": classifier = KNeighborsClassifier(n_neighbors=3)
            elif alg == "rf": classifier = RandomForestClassifier(n_estimators=10, max_depth=12, class_weight=balanced)
            classifier = classifier.fit(examples, labels)
            pred_results_train, pred_results_val = self.mc_predict_classes(classifier)
            results_train.append(pred_results_train)
            results_val.append(pred_results_val)
        fname_train, fname_val = self.get_fname(alg, None, balanced)
        print("Saving results to: {}, {}".format(fname_train, fname_val))
        self.save_results(results_train, fname_train)
        self.save_results(results_val, fname_val)

    def get_vectors(self, c):
        if c == "texas":
            return self.texas_train, self.texas_labels_train, self.texas_val, self.texas_labels_val
        if c == "france":
            return self.france_train, self.france_labels_train, self.france_val, self.france_labels_val
        if c == "maine":
            return self.maine_train, self.maine_labels_train, self.maine_val, self.maine_labels_val
        if c == "oregon":
            return self.oregon_train, self.oregon_labels_train, self.oregon_val, self.oregon_labels_val

    def two_class_predict_classes(self, classifier, c1_train, c1_val, c2_train, c2_val):
        c1_res_train = self.binary_predict(classifier, c1_train, True)
        c1_res_val = self.binary_predict(classifier, c1_val, True)
        c2_res_train = self.binary_predict(classifier, c2_train, False)
        c2_res_val = self.binary_predict(classifier, c2_val, False)
        return [c1_res_train, c2_res_train], [c1_res_val, c2_res_val]

    def two_class(self, alg, c1, c2, balanced):
        print("Running TWO-CLASS: Alg={}, C1={}, C2={}, Balanced={}".format(alg, c1, c2, balanced))
        self.binary_labels(c1)
        c1_train, c1_train_labels, c1_val, c1_val_labels = self.get_vectors(c1)
        c2_train, c2_train_labels, c2_val, c2_val_labels = self.get_vectors(c2)
        examples = np.concatenate((c1_train, c2_train)) 
        labels = np.concatenate((c1_train_labels, c2_train_labels))        
        results_train = []
        results_val = []
        for i in range(50):
            if alg == "sgd": classifier = SGDClassifier(class_weight=balanced)
            elif alg == "lr": classifier = LogisticRegression(class_weight=balanced)
            elif alg == "knn": classifier = KNeighborsClassifier(n_neighbors=3)
            elif alg == "rf": classifier = RandomForestClassifier(n_estimators=10, max_depth=12, class_weight=balanced)
            classifier = classifier.fit(examples, labels)
            score = classifier.score(examples, labels)
            pred_results_train, pred_results_val = self.two_class_predict_classes(classifier, c1_train, c1_val, c2_train, c2_val)
            results_train.append(pred_results_train)
            results_val.append(pred_results_val)
        fname_train, fname_val = self.get_fname(alg+"2", c1, balanced)
        print("Saving results to: {}, {}".format(fname_train, fname_val))
        self.save_results(results_train, fname_train)
        self.save_results(results_val, fname_val)

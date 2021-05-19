import csv
import numpy as np

###############
## Loads raw results and agglomerates them
###############

class Resulter():
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def one_vs_all(self, alg, balanced):
        if balanced: 
            b = "_b"
        else:
            b = ""
        fnames_train = [alg+"_texas"+b+"_train.csv", alg+"_france"+b+"_train.csv", alg+"_maine"+b+"_train.csv", alg+"_oregon"+b+"_train.csv"]
        fnames_val = [alg+"_texas"+b+"_val.csv", alg+"_france"+b+"_val.csv", alg+"_maine"+b+"_val.csv", alg+"_oregon"+b+"_val.csv"]
        
        results_train = []
        for fname in fnames_train:
            with open(self.result_dir+fname, "r") as f:
                all_res = []
                reader = csv.reader(f)
                for row in reader:
                    all_res.append(list(map(float, row)))
            averages = np.mean(all_res, axis=0)
            averages = [round(val, 3) for val in averages]
            results_train.append(averages)
        results_val = []
        for fname in fnames_val:
            with open(self.result_dir+fname, "r") as f:
                all_res = []
                reader = csv.reader(f)
                for row in reader:
                    all_res.append(list(map(float, row)))
            averages = np.mean(all_res, axis=0)
            averages = [round(val, 3) for val in averages]
            results_val.append(averages)

        with open(self.result_dir+"summaries/"+alg+b+"_ova.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Train"])
            writer.writerow(["", "Paris, TX", "Paris, FR", "Portland, ME", "Portland, OR"])
            writer.writerow(["Paris, TX", results_train[0][0], results_train[0][1], results_train[0][2], results_train[0][3]])
            writer.writerow(["Paris, FR", results_train[1][0], results_train[1][1], results_train[1][2], results_train[1][3]])
            writer.writerow(["Portland, ME", results_train[2][0], results_train[2][1], results_train[2][2], results_train[2][3]])
            writer.writerow(["Portland, OR", results_train[3][0], results_train[3][1], results_train[3][2], results_train[3][3]])
            writer.writerow(["Validation"])
            writer.writerow(["", "Paris, TX", "Paris, FR", "Portland, ME", "Portland, OR"])
            writer.writerow(["Paris, TX", results_val[0][0], results_val[0][1], results_val[0][2], results_val[0][3]])
            writer.writerow(["Paris, FR", results_val[1][0], results_val[1][1], results_val[1][2], results_val[1][3]])
            writer.writerow(["Portland, ME", results_val[2][0], results_val[2][1], results_val[2][2], results_val[2][3]])
            writer.writerow(["Portland, OR", results_val[3][0], results_val[3][1], results_val[3][2], results_val[3][3]])
            
    def multi_class(self, alg, balanced):
        if balanced: 
            b = "_b"
        else:
            b = ""
        fname_train = alg+b+"_train.csv"
        fname_val = alg+b+"_val.csv"
        
        with open(self.result_dir+fname_train, "r") as f:
            results = []
            reader = csv.reader(f)
            for row in reader:
                clean_row = []
                for l in row:
                    l = l[1:-1].split(",")
                    clean_row.append(list(map(int, l)))
                results.append(clean_row)
        averages = np.mean(results, axis=0)
        results_train = [[round(val, 3) for val in row] for row in averages]

        with open(self.result_dir+fname_val, "r") as f:
            results = []
            reader = csv.reader(f)
            for row in reader:
                clean_row = []
                for l in row:
                    l = l[1:-1].split(",")
                    clean_row.append(list(map(int, l)))
                results.append(clean_row)
        averages = np.mean(results, axis=0)
        results_val = [[round(val, 3) for val in row] for row in averages]
        
        with open(self.result_dir+"summaries/"+alg+b+"_mc.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Train"])
            writer.writerow(["", "Paris, TX", "Paris, FR", "Portland, ME", "Portland, OR"])
            writer.writerow(["Paris, TX", results_train[0][0], results_train[0][1], results_train[0][2], results_train[0][3]])
            writer.writerow(["Paris, FR", results_train[1][0], results_train[1][1], results_train[1][2], results_train[1][3]])
            writer.writerow(["Portland, ME", results_train[2][0], results_train[2][1], results_train[2][2], results_train[2][3]])
            writer.writerow(["Portland, OR", results_train[3][0], results_train[3][1], results_train[3][2], results_train[3][3]])
            writer.writerow(["Validation"])
            writer.writerow(["", "Paris, TX", "Paris, FR", "Portland, ME", "Portland, OR"])
            writer.writerow(["Paris, TX", results_val[0][0], results_val[0][1], results_val[0][2], results_val[0][3]])
            writer.writerow(["Paris, FR", results_val[1][0], results_val[1][1], results_val[1][2], results_val[1][3]])
            writer.writerow(["Portland, ME", results_val[2][0], results_val[2][1], results_val[2][2], results_val[2][3]])
            writer.writerow(["Portland, OR", results_val[3][0], results_val[3][1], results_val[3][2], results_val[3][3]])
        
    def two_class_paris(self, alg, balanced):
        if balanced: 
            b = "_b"
        else:
            b = ""
        fnames_train = [alg+"2_texas"+b+"_train.csv", alg+"2_france"+b+"_train.csv"]
        fnames_val = [alg+"2_texas"+b+"_val.csv", alg+"2_france"+b+"_val.csv"]
        print(fnames_train)

        results_train = []
        for fname in fnames_train:
            all_res = []
            with open(self.result_dir+fname, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    all_res.append(list(map(float, row)))
            averages = np.mean(all_res, axis=0)
            averages = [round(val, 3) for val in averages]
            results_train.append(averages)
        print(results_train)

        results_val = []
        for fname in fnames_val:
            all_res = []
            with open(self.result_dir+fname, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    all_res.append(list(map(float, row)))
            averages = np.mean(all_res, axis=0)
            averages = [round(val, 3) for val in averages]
            results_val.append(averages)
        print(results_val)

        with open(self.result_dir+"summaries/"+alg+b+"_paris.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Train"])
            writer.writerow(["", "Paris, TX", "Paris, FR"])
            writer.writerow(["Paris, TX", results_train[0][0], results_train[0][1]])
            writer.writerow(["Paris, FR", results_train[1][1], results_train[1][0]])
            writer.writerow(["Validation"])
            writer.writerow(["", "Paris, TX", "Paris, FR"])
            writer.writerow(["Paris, TX", results_val[0][0], results_val[0][1]])
            writer.writerow(["Paris, FR", results_val[1][1], results_val[1][0]])

    def two_class_portland(self, alg, balanced):     
        if balanced: 
            b = "_b"
        else:
            b = ""
        fnames_train = [alg+"2_maine"+b+"_train.csv", alg+"2_oregon"+b+"_train.csv"]
        fnames_val = [alg+"2_maine"+b+"_val.csv", alg+"2_oregon"+b+"_val.csv"]
        print(fnames_train)

        results_train = []
        for fname in fnames_train:
            all_res = []
            with open(self.result_dir+fname, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    all_res.append(list(map(float, row)))
            averages = np.mean(all_res, axis=0)
            averages = [round(val, 3) for val in averages]
            results_train.append(averages)
        print(results_train)

        results_val = []
        for fname in fnames_val:
            all_res = []
            with open(self.result_dir+fname, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    all_res.append(list(map(float, row)))
            averages = np.mean(all_res, axis=0)
            averages = [round(val, 3) for val in averages]
            results_val.append(averages)
        print(results_val)

        with open(self.result_dir+"summaries/"+alg+b+"_portland.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Train"])
            writer.writerow(["", "Portland, ME", "Portland, OR"])
            writer.writerow(["Portland, ME", results_train[0][0], results_train[0][1]])
            writer.writerow(["Portland, OR", results_train[1][1], results_train[1][0]])
            writer.writerow(["Validation"])
            writer.writerow(["", "Portland, ME", "Portland, OR"])
            writer.writerow(["Portland, ME", results_val[0][0], results_val[0][1]])
            writer.writerow(["Portland, OR", results_val[1][1], results_val[1][0]])
      
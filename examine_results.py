import numpy as np
import csv

results = []
with open("results/rf2_france_b.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 0: continue
        print(row)
        results.append(list(map(float, row)))
        # clean_row = []
        # for l in row:
        #     l = l[1:-1].split(",")
        #     clean_row.append(list(map(int, l)))
        # results.append(clean_row)

# exit()
means = np.mean(results, axis=0)
# medians = np.median(results, axis=0)
print(means)
# print(medians)
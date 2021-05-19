import numpy as np
import json

################
## Convert dump files into json
################

data = []

with open("data/portland_or.dump", "r", encoding="utf8") as f:
    raw_data = f.readlines()
raw_data = "".join(raw_data).replace("\n", "")

data = raw_data.split("#######################")
data = [row.split("**************") for row in data]
data = data[:500]

data_dict = {}

for row in data:
    if len(row) < 2: continue
    data_dict[row[0]] = row[1]

with open("data/portland_or_500.json", "w") as f:
    json.dump(data_dict, f)
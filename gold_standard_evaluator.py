import csv
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ANS_FILE_LIST = ["SVM_1561109304.csv"]
ANS_FILE_DIR = "answers/"

GOLD_STANDARD_FILE = ""

if __name__ == "__main__":
    sense_counter = defaultdict(int)
    with open(ANS_FILE_DIR+ANS_FILE_LIST[0], "r") as f:
        data = csv.reader(f)
        for row in data:
            sense_counter[row[2]]+= 1
    print(sense_counter)
    for sense in list(sense_counter.keys()):
        print(sense, sense_counter[sense])
    print(sorted(list(sense_counter.keys())))

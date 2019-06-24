from datetime import timedelta
from collections import defaultdict

import csv
import numpy as np
import time

def open_dataset(file_list, ret_dict=True):
    dataset = {} if ret_dict else []
    for file_name in file_list:
        if ret_dict:
            dataset[file_name] = open_data_file(file_name)
        else:
            dataset += open_data_file(file_name)
    return dataset

def open_data_file(file_name):
    dataset = []
    with open(file_name, "r") as csv_file:
        csv_data = csv.DictReader(csv_file)
        for row in csv_data:
            row["sense"] = row["sense"][:4]
            if row["sense"] == "----":
                continue
            dataset.append(row)
    return dataset

def disagreement_handling(dataset):
    clean_sentence = {}
    for data in dataset:
        if data["\ufeffkalimat_id"] in clean_sentence:
            current_data = clean_sentence[data["\ufeffkalimat_id"]]
            if data["freq"] < current_data["freq"]:
                continue
            if data["sense"] == current_data["sense"]:
                if len(data["kalimat"]) < len(current_data["kalimat"]):
                    continue
        clean_sentence[data["\ufeffkalimat_id"]] = data
    return clean_sentence.values()

def analyze_data(data):
    sense_counter = defaultdict(int)
    word_counter = defaultdict(int)
    for datum in data:
        word_counter[datum["kata"]] += 1
        sense_counter[datum["sense"]] += 1
        # sense.split("")
    print("Total Sentence : {}".format(len(data)))
    print("Average Sense : {}".format(len(data)/len(sense_counter)))
    print("Max Sense Counter : {}".format(max(list(sense_counter.values()))))
    print("Min Sense Counter : {}".format(min(list(sense_counter.values()))))
    print("Sense Counter :")
    print(sense_counter)
    for sense in sorted(list(sense_counter.keys())):
        print(sense, sense_counter[sense])
    print(sorted(list(sense_counter.keys())))
    print("Word Counter :")
    print(word_counter)

def demo():
    pass

if __name__ == "__main__":
    demo()

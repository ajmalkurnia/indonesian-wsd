import csv
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

from data import open_dataset, disagreement_handling, analyze_data
from pre_processing import Preprocess
from feature_extraction import Features
from log import Log

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


TRAINING_FILES = ["single_annotator.csv",
                "double_annotator_agree.csv", "double_annotator_disagree.csv",
                "triple_annotator_agree.csv", "triple_annotator_disagree.csv"]
TRAINING_DIR = "training_set/"
SENSE_FILES = "55WordSenses.xml"
TESTING_FILE = "testing_data.csv"

log = Log()
def build_dataset(merged_dataset):
    word_matrix = dict()
    for data in merged_dataset:
        if "data_embedding" in data:
            if data["kata"] in word_matrix:
                word_matrix[data["kata"]][0].append(data["data_embedding"])
                word_matrix[data["kata"]][1].append(data["sense"])
            else:
                word_matrix[data["kata"]] = [[data["data_embedding"]], [data["sense"]]]

    sense_counter = defaultdict(int)
    for data in merged_dataset:
        if "data_embedding" in data:
            sense_counter[data["sense"]] += 1
    for word, data in word_matrix.items():
        sense_set = set()
        for idx, sense in enumerate(data[1]):
            sense_set.add(sense)
        for sense in sense_set:
            sense_index = []
            for idx, ss in enumerate(data[1]):
                if ss == sense:
                    sense_index.append(idx)
            # Duplicate data with low count of sense
            if len(sense_index) < 30:
                random_index = np.random.choice(sense_index, size=30-len(sense_index))
                for i in random_index:
                    data[0].append(data[0][i])
                    data[1].append(data[1][i])
            # elif len(sense_index) > 100:
            #     random_index = np.random.choice(sense_index, size=len(sense_index)-80)
            #     # Down sample
            #     for i in sorted(random_index, reverse=True):
            #         del data[0][i]
            #         del data[1][i]
    # print(word_matrix["jam"][1])
    word_matrix_train = dict()
    word_matrix_test = dict()
    for word, data in word_matrix.items():
        data[0] = np.array(data[0])
        train_x, test_x, train_y, test_y = train_test_split(data[0], data[1], test_size=0.1, random_state=491238)
        word_matrix_train[word] = [train_x, train_y]
        word_matrix_test[word] = [test_x, test_y]
    return word_matrix, word_matrix_train, word_matrix_test

def prepare_test_data():
    test_data = []
    log.write("Open test set")
    with open(TESTING_FILE, "r") as csv_file:
        csv_data = csv.DictReader(csv_file)
        for row in csv_data:
            test_data.append(row)
    log.write("Preprocessed test set")
    for data in test_data:
        data["preprocessed_kalimat"] = Preprocess(data).preprocess()
    log.write("Extract feature test set")
    feature = Features(test_data)
    feature.extract_feature()
    # feature.get_trainable_dataset()
    return test_data

def actual_test(test_data, model, word, ansfile):
    # log.write("Model Parameter")
    # log.write(model.get_params())
    # log.write("Build data matrix")
    feature_matrix = []
    tested_data = []

    for data in test_data:
        # print(word, data["word"])
        if word == data["word"]:
            tested_data.append(data["\ufeffid"])
            feature_matrix.append(data["data_embedding"])
    feature_matrix = np.array(feature_matrix)
    # log.write("Predict test set")
    # print(feature_matrix.shape)
    # print(feature_matrix)
    prediction = model.predict(feature_matrix)
    # log.write("Write test result")
    if not os.path.exists("answers"): os.makedirs("answers")
    with open(ansfile, "a") as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx, data in enumerate(tested_data):
            csv_writer.writerow([data, word, prediction[idx]])


def main():
    log.write("Open dataset")
    dataset = open_dataset([TRAINING_DIR+files for files in TRAINING_FILES])
    merged_dataset = []
    log.write("Resolve disagreement data")
    for k, v in dataset.items():
        if k == TRAINING_DIR+TRAINING_FILES[2] or k == TRAINING_DIR+TRAINING_FILES[4]:
            dataset[k] = disagreement_handling(v)
        merged_dataset += dataset[k]
    analyze_data(merged_dataset)
    log.write("Analyzing sense")
    sense_id = set()
    for datum in merged_dataset:
        sense_id.add(datum["sense"])
    xml_root = ET.parse(SENSE_FILES).getroot()
    for word in xml_root:
        for sense in word.findall("senses/sense"):
            if word.attrib["wid"].zfill(2)+sense.attrib["sid"].zfill(2) not in sense_id:
                log.write("Kata `{}` dengan sense `{}` tidak ditemukan di data training".format(word[0].text, sense.attrib))
    log.write("Preprocessing")
    for data in merged_dataset:
        data["preprocessed_kalimat"] = Preprocess(data).preprocess()

    # for datum in merged_dataset:
    #     datum["preprocessed_kalimat"] =

    print(merged_dataset[0])
    log.write("Feature extraction")
    feature = Features(merged_dataset)
    feature.extract_feature()
    # feature.get_trainable_dataset()
    # log.write(merged_dataset[0])
    with open("feature.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        # csv_writer.writerow(["kalimat_id", "sense", "features"])
        for data in merged_dataset:
            if "data_embedding" in data:
                csv_writer.writerow([data["\ufeffkalimat_id"], data["kata"], data["sense"]]+list(data["data_embedding"]))
    log.write("Build Dataset")
    word_feature_mat, dummy_train, dummy_test = build_dataset(merged_dataset)
    classifier = {
                "Random Forest":RandomForestClassifier(n_estimators=1000),
                "SVM":SVC(C=10000, gamma=0.1, tol=1e-6, decision_function_shape='ovo'),
                "Neural Net":MLPClassifier(hidden_layer_sizes=2000, activation='tanh',
                        solver='adam', tol=1e-6, learning_rate_init=0.001, max_iter=1000, early_stopping=True)
                }
    best_model = None
    best_acc = 0.0000001
    test_data = prepare_test_data()
    for model_name, model_class in classifier.items():
        log.write("Try {} :".format(model_name))
        true_count = 0
        n_data = 0
        model = model_class
        ansfile = "answers/{}_{}.csv".format(model_name, int(time.time()))
        for word in sorted(list(word_feature_mat.keys())):
            print("predicting {}".format(word))
            model.fit(dummy_train[word][0], dummy_train[word][1])
            prediction = model.predict(dummy_test[word][0])
            n_data += len(prediction)
            for pred, true in zip(prediction, dummy_test[word][1]):
                if pred == true:
                    true_count += 1
            model = model.fit(word_feature_mat[word][0], word_feature_mat[word][1])
            actual_test(test_data, model, word, ansfile)
        accuracy = 100*true_count/n_data
        # if accuracy > best_acc:
    #     #     best_model = model_class
        log.write("Akurasi dari {} : {} %".format(model_name, accuracy))
    #     log.write("Train {} model using all train_data".format(model_name))

if __name__ == "__main__":
    main()

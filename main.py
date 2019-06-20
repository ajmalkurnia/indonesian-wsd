import csv
import os
import time
import xml.etree.ElementTree as ET

from data import open_dataset, disagreement_handling, analyze_data
from pre_processing import Preprocess
from feature_extraction import Features
# from evaluation import Evaluator
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
    feature_matrix = []
    labels = []
    for data in merged_dataset:
        if "data_embedding" in data:
            labels.append(data["sense"])
            feature_matrix.append(data["data_embedding"])
    feature_matrix = np.array(feature_matrix)
    train_data, test_data, train_label, test_label = train_test_split(feature_matrix,
                                                labels, test_size=0.2, random_state=8132)
    return [train_data, train_label], [test_data, test_label]

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
    feature.get_trainable_dataset()
    return test_data

def actual_test(test_data, model, model_name):
    log.write("Model Parameter")
    log.write(model.get_params())
    log.write("Build data matrix")
    feature_matrix = []
    for data in test_data:
        feature_matrix.append(data["data_embedding"])
    feature_matrix = np.array(feature_matrix)
    log.write("Predict test set")
    prediction = model.predict(feature_matrix)
    log.write("Write test result")
    if not os.path.exists("answers"): os.makedirs("answers")
    ansfile = "answers/{}_{}.csv".format(model_name, int(time.time()))
    with open(ansfile, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx, data in enumerate(test_data):
            csv_writer.writerow([data["\ufeffid"], data["word"], prediction[idx]])


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
    for datum in merged_dataset:
        datum["preprocessed_kalimat"] = Preprocess(datum).preprocess()

    log.write("Feature extraction")
    feature = Features(merged_dataset)
    feature.extract_feature()
    feature.get_trainable_dataset()
    # log.write(merged_dataset[0])
    log.write("Build Dataset")
    train, dummy_test = build_dataset(merged_dataset)
    log.write("Dimension")
    log.write(train[0].shape)

    classifier = {
                "Random Forest":RandomForestClassifier(n_estimators=1000),
                "SVM":SVC(C=100, gamma=0.001, tol=1e-6, decision_function_shape='ovo'),
                "Neural Net":MLPClassifier(hidden_layer_sizes=1400, activation='logistic',
                        solver='adam', tol=1e-6, learning_rate_init=0.001, max_iter=1000, early_stopping=True)
                }
    best_model = None
    best_acc = 0.0000001
    test_data = prepare_test_data()
    for model_name, model_class in classifier.items():
        log.write("Try {} :".format(model_name))
        model = model_class
        model.fit(train[0], train[1])
        prediction = model.predict(dummy_test[0])
        true_count = 0
        for pred, true in zip(prediction, dummy_test[1]):
            if pred == true:
                true_count += 1
        accuracy = 100*true_count/len(prediction)
        # if accuracy > best_acc:
        #     best_model = model_class
        log.write("Akurasi dari {} : {} %".format(model_name, accuracy))

        log.write("Train {} model using best model using all train_data".format(model_name))
        model = model.fit(np.vstack((train[0], dummy_test[0])), np.concatenate([train[1], dummy_test[1]]))
        actual_test(test_data, model, model_name)

if __name__ == "__main__":
    main()

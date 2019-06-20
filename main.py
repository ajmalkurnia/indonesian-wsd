import csv
import xml.etree.ElementTree as ET

from data import open_dataset, disagreement_handling, analyze_data
from pre_processing import Preprocess
from feature_extraction import Features
from evaluation import Evaluator

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

def build_dataset(merged_dataset):
    feature_matrix = []
    labels = []
    for data in merged_dataset:
        if "data_embedding" in data:
            labels.append(data["sense"])
            feature_matrix.append(data["data_embedding"])
    feature_matrix = np.array(feature_matrix)
    train_data, test_data, train_label, test_label = train_test_split(feature_matrix,
                                                labels, test_size=0.33, random_state=8132)
    return [train_data, train_label], [test_data, test_label]


def main():
    print("Open dataset")
    dataset = open_dataset([TRAINING_DIR+files for files in TRAINING_FILES])
    merged_dataset = []
    print("Resolve disagreement data")
    for k, v in dataset.items():
        if k == TRAINING_DIR+TRAINING_FILES[2] or k == TRAINING_DIR+TRAINING_FILES[4]:
            dataset[k] = disagreement_handling(v)
        merged_dataset += dataset[k]
    analyze_data(merged_dataset)
    sense_id = set()
    for datum in merged_dataset:
        sense_id.add(datum["sense"])
    xml_root = ET.parse(SENSE_FILES).getroot()
    for word in xml_root:
        for sense in word.findall("senses/sense"):
            if word.attrib["wid"].zfill(2)+sense.attrib["sid"].zfill(2) not in sense_id:
                print("Kata `{}` dengan sense `{}` tidak ditemukan di data training".format(word[0].text, sense.attrib))
    print("Preprocessing")
    for datum in merged_dataset:
        datum["preprocessed_kalimat"] = Preprocess(datum).preprocess()

    print("Feature extraction")
    feature = Features(merged_dataset)
    feature.extract_feature()
    feature.get_trainable_dataset()
    print(merged_dataset[0])
    train, dummy_test = build_dataset(merged_dataset)
    print(train[0].shape, dummy_test[0].shape)

    classifier = {"SVM":SVC(C=0.1, gamma='auto', tol=1e-6, decision_function_shape='ovo'),
                "Random Forest":RandomForestClassifier(n_estimators=100),
                "Neural Net":MLPClassifier(hidden_layer_sizes=1400, activation='logistic',
                        solver='lbfgs')
                }
    for model_name, model_class in classifier.items():
        model = model_class
        model.fit(train[0], train[1])
        prediction = model.predict(dummy_test[0])
        true_count = 0
        for pred, true in zip(prediction, dummy_test[1]):
            if pred == true:
                true_count += 1
        print("Akurasi dari {} : {} %".format(model_name, 100*true_count/len(prediction)))


if __name__ == "__main__":
    main()

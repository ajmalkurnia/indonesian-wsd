import csv
import xml.etree.ElementTree as ET
from data import open_dataset, disagreement_handling, analyze_data
from pre_processing import Preprocess
from feature_extraction import Features
TRAINING_FILES = ["single_annotator.csv",
                "double_annotator_agree.csv", "double_annotator_disagree.csv",
                "triple_annotator_agree.csv", "triple_annotator_disagree.csv"]
TRAINING_DIR = "training_set/"
SENSE_FILES = "55WordSenses.xml"
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
        # print(word.tag, word.attrib)
        for sense in word.findall("senses/sense"):
            # print(sense.tag, sense.attrib)
            if word.attrib["wid"].zfill(2)+sense.attrib["sid"].zfill(2) not in sense_id:
                print("Kata `{}` dengan sense `{}` tidak ditemukan di data training".format(word[0].text, sense.attrib))
    print("Preprocessing")
    # lol = set()
    # error_word = 0
    for datum in merged_dataset:
        datum["preprocessed_kalimat"] = Preprocess(datum).preprocess()
        # if datum["kata"] not in datum["preprocessed_kalimat"]:
        #     error_word += 1
            # print("##########################################")
            # print(datum["\ufeffkalimat_id"])
            # print(datum["kata"])
            # print(datum["kalimat"])
            # print(datum["preprocessed_kalimat"])
    # print(error_word)
    print("Feature extraction")
    feature = Features(merged_dataset)
    feature.extract_feature()
    feature.get_trainable_dataset()
        # for grand_child in child:

    # print(feature.__dataset[0])
    # print(merged_dataset[2])
    # for data in merge_dataset:
    #

if __name__ == "__main__":
    main()

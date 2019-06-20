from datetime import timedelta
from collections import defaultdict
from gensim.models import word2vec, Word2Vec
from nltk.tag import CRFTagger
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import csv
import numpy as np
import time
import gensim
import multiprocessing

W2V_MODEL_FILE = [""]
POSTAG_MODEL_DIR = "external_source/all_indo_man_tag_corpus_model.crf.tagger"
PRECOMPUTED_W2V_DIR = "external_source/"
CORPORA_W2V_DIR = "external_source/idwiki.txt"
EMBEDDING_SIZE = 80

class Features:

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__postag = set()

    def __load_precomputed_w2v_model(model_directory):
        word_embedding = Word2Vec.load(model_directory)
        # print(word_embedding.layer1_size)
        return word_embedding

    def word_embedding_training(self, pretraining=True, pre_train_dir=""):
        new_sentences = [data["preprocessed_kalimat"] for data in self.__dataset]
        if pre_train_dir:
            self.__embedding_model = __load_precomputed_w2v_model(pre_train_dir)
            self.__embedding_model.build_vocab(new_sentences, update=True)
            self.__embedding_model.train(new_sentences, total_examples=emb_model.corpus_count, epochs=emb_model.iter)
        elif pretraining:
            sentences = word2vec.LineSentence(CORPORA_W2V_DIR)
            self.__embedding_model = word2vec.Word2Vec(sentences, size=EMBEDDING_SIZE, workers=multiprocessing.cpu_count()-1, min_count=0)
            self.__embedding_model.build_vocab(new_sentences, update=True)
            self.__embedding_model.train(new_sentences, total_examples=self.__embedding_model.corpus_count, epochs=self.__embedding_model.iter)
        else:
            self.__embedding_model = word2vec.Word2Vec(new_sentences, size=EMBEDDING_SIZE, workers=multiprocessing.cpu_count()-1)

    def postag_sequence(self, data):
        ct = CRFTagger()
        ct.set_model_file(POSTAG_MODEL_DIR)
        data["postag_seq"] = ct.tag_sents([data["preprocessed_kalimat"]])[0]
        for datum in data["postag_seq"]:
            self.__postag.add(datum[1])

    def get_trainable_dataset(self):
        self.set_up_postag_embedding()
        for data in self.__dataset:
            postag_embedding = self.get_postag_embedding(data)
            word_embedding = self.get_word_embedding(data)
            postag_dim = postag_embedding.shape[1]
            word_dim = len(word_embedding[0])
            try:
                word_index = data["preprocessed_kalimat"].index(data["kata"])
            except ValueError:
                word_found = False
                for idx, kata in enumerate(data["preprocessed_kalimat"]):
                    if data["kata"] in kata:
                        word_found = True
                        word_index = idx
                if not word_found:
                    print("Kalimat Id {} tidak dapat digunakan karena kata tidak ditemukan pada kalimat tersebut".format(data["\ufeffkalimat_id"]))
                    continue
            data_embedding = []
            for i in range(word_index-3, word_index+4):
                if word_index < 0 or word_index >= len(data["preprocessed_kalimat"]):
                    data_embedding.append(np.zeros(postag_dim+word_dim))
                    # data_embedding.append(np.zeros(word_dim))
                else:
                    data_embedding.append(np.concatenate([word_embedding[word_index], postag_embedding[word_index]]))
                    # data_embedding.append(word_embedding[word_index])
            data["data_embedding"] = np.concatenate(data_embedding)
            # print(np.concatenate(this_word_embedding).shape)
            # Merge embedding
            # Word selection
            # Sense encoding
            #

    def set_up_postag_embedding(self):
        self.__label_encoder = LabelEncoder().fit(list(self.__postag))
        integer_encoded = self.__label_encoder.transform(list(self.__postag))
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.__onehot_encoder = OneHotEncoder(sparse=False).fit(integer_encoded)

    def get_postag_embedding(self, data):
        postag_data = [datum[1] for datum in data["postag_seq"]]
        integer_encoded = self.__label_encoder.transform(postag_data)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return self.__onehot_encoder.transform(integer_encoded)

    def get_word_embedding(self, data):
        embedding_sequence = []
        for word in data["preprocessed_kalimat"]:
            embedding_sequence.append(self.__embedding_model[word])
        return embedding_sequence

    def extract_feature(self):
        self.word_embedding_training(pretraining=True)
        for data in self.__dataset:
            self.postag_sequence(data)
        self.get_trainable_dataset()

def demo():
    pass

if __name__ == "__main__":
    demo()

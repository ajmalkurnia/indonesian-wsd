from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from spacy.lemmatizer import Lemmatizer
from spacy.lang.id import LOOKUP
import string
import nltk

class Preprocess:

    __sentence = ""

    def __init__(self,data):
        self.__sentence = data["kalimat"]
        self.word = data["kata"] if "kata" in data else data["word"]

    def tokenization(self):
        self.__sentence = nltk.wordpunct_tokenize(self.__sentence)

    def remove_stopword(self):
        # 1remove stop word
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        self.__sentence = [stopword.remove(word) if word != self.word else word for word in self.__sentence]

    def stemming(self):
        # 2. stemmer==================================================
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        # print(self.__sentence)
        self.__sentence = [stemmer.stem(word) if word != self.word else word for word in self.__sentence]

    def lemmatization(self):
        lemmatizer = Lemmatizer(lookup=LOOKUP)
        # lemma = []
        for word in self.__sentence:
            word = lemmatizer(word, "NOUN")
        # self.__sentence = " ".join(lemma)

    def clean_sentence(self):
        # 3.punctuation===================-================
        table = str.maketrans({key: None for key in string.punctuation})
        self.__sentence = [word.translate(table) for word in self.__sentence]

        # remove digits
        remove_digits = str.maketrans('', '', string.digits)
        self.__sentence = [word.translate(remove_digits) for word in self.__sentence]

        # 4.remove double space===============================
        # self.__sentence = ' '.join(self.__sentence.split()

        #5. remove single letter word
        self.__sentence = [w.lower() for w in self.__sentence if len(w) > 1]

        exception = {"senilai": "nilai", "bernilai": "nilai", "menilainya":"nilai",
                    "dinilainya":"nilai","penilain": "penilaian",
                    "menurunkannya": "menurunkan","mengeluarkannya":"mengeluarkan"}

        self.__sentence = [exception.get(word, word) for word in self.__sentence]

    def preprocess(self):
        self.tokenization()
        # self.remove_stopword()
        self.clean_sentence()
        # self.lemmatization()
        self.stemming()
        return self.__sentence
'''
kalimat = "saya makan buah 500,/s., dengan nya"
budi =Preprocess(kalimat).preprocess()
print(budi)
'''

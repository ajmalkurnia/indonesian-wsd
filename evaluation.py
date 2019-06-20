class Evaluator:
    def __init__(self):
        self.confusion_matrix = {(i,j): 0 for i in range(2) for j in range(2)}
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self._n_data = 0

    def _make_confusion_matrix(self, truth, predicted):
        assert len(truth) == len(predicted)
        self._n_data = len(truth)
        for idx in range(len(truth)):
            self.confusion_matrix[(truth[idx]), (predicted[idx])] += 1

    def _accuracy(self):
        tp = self.confusion_matrix[(1,1)]
        tn = self.confusion_matrix[(0,0)]
        self.accuracy = 1.0*(tp+tn)/self._n_data

    def _precision(self):
        tp = self.confusion_matrix[(1,1)]
        fp = self.confusion_matrix[(0,1)]
        self.precision = 1.0*(tp)/(tp+fp)

    def _recall(self):
        tp = self.confusion_matrix[(1,1)]
        fn = self.confusion_matrix[(1,0)]
        self.recall = 1.0*(tp)/(tp+fn)

    def _f1_score(self):
        self.f1_score = 2/((1/self.recall)+(1/self.precision))

    def compute_all(self, truth, predicted):
        self._make_confusion_matrix(truth, predicted)
        self._accuracy()
        self._precision()
        self._recall()
        self._f1_score()

import csv
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib

class AspectExtractor(object):

    TARGET_NAMES = ['cleanliness', 'food/drinks', 'location', 'room amenities', 'staff']

    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.pipeline = Pipeline([
            ('features', FeatureUnion(
                transformer_list = [
                    ('bag_of_ngram', CountVectorizer(ngram_range=(1, 2)))
                ]
            )),
            ('clf', OneVsRestClassifier(LogisticRegression()))
        ])

    @classmethod
    def read_data(cls, filename):
        """
        Load dataset from csv.

        Parameters
        ----------
        filename: Filename of the dataset in csv.

        Returns
        -------
        data: Review sentences.
        targets: data labels (aspects).
        """

        data = []
        targets = []
        regex = re.compile('[^0-9a-zA-Z]+')

        with open (filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';', quotechar='"')
            next(reader)
            for row in reader:
                text = regex.sub(' ', row[0])
                data.append(text)
                target = []
                for i in range(1, len(cls.TARGET_NAMES) + 1):
                    if (row[i] == '1'):
                        target.append(cls.TARGET_NAMES[i - 1])
                targets.append(target)
        return data, targets

    def fit(self, X, y):
        """
        Train model.

        Parameters
        ----------
        X: Train data.
        y: Train data labels.

        Returns
        -------
        self
        """

        X = np.array(X)
        labels = self.mlb.fit_transform(y)
        self.clf = self.pipeline.fit(X, labels)
        return self

    def predict(self, X):
        """
        Predict the given data.

        Parameters
        ----------
        X: Data to predict.

        Returns
        -------
        y: labels for the given data.
        """

        X = np.array(X)
        labels = self.clf.predict(X)
        results = []
        for i in range(len(X)):
            result = {}
            result['sentence'] = X[i]
            result['aspects'] = []
            for j in range(len(self.mlb.classes_)):
                if labels[i][j] == 1:
                    result['aspects'].append(self.mlb.classes_[j])
            results.append(result)
        return results

    def cross_validate(self, X, y, k):
        """
        KFold cross validation.

        Parameters
        ----------
        X: Train data.
        y: Train data labels.
        k: Number of folds.
        """

        X = np.array(X)
        labels = self.mlb.fit_transform(y)

        precision_scores = [[] for i in range(len(self.mlb.classes_))]
        recall_scores =  [[] for i in range(len(self.mlb.classes_))]
        f1_scores =  [[] for i in range(len(self.mlb.classes_))]
        kf = KFold(n_splits=k)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            clf = self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_test)

            precision = precision_score(y_test, y_pred, average=None)
            recall = recall_score(y_test, y_pred, average=None)
            f1 = f1_score(y_test, y_pred, average=None)

            for i in range(len(self.mlb.classes_)):
                precision_scores[i].append(precision[i])
                recall_scores[i].append(recall[i])
                f1_scores[i].append(f1[i])

        print("Cross validation results:")
        for i in range(len(self.mlb.classes_)):
            print("\tCategory:", self.mlb.classes_[i])
            print("\t\tPrecision:", np.array(precision_scores[i]).mean())
            print("\t\tRecall:", np.array(recall_scores[i]).mean())
            print("\t\tF1-score:", np.array(f1_scores[i]).mean())

    def evaluate(self, test_data_filename):
        """
        Evaluate the trained model using test data.

        Parameters
        ----------
        test_data_filename: Filename for the test data.
        """

        X, y = AspectExtractor.read_data(test_data_filename)
        labels = self.mlb.transform(y)
        y_pred = self.clf.predict(X)

        precision = np.array(precision_score(labels, y_pred, average=None))
        recall = np.array(recall_score(labels, y_pred, average=None))
        f1 = np.array(f1_score(labels, y_pred, average=None))

        print("Evaluation results:")
        for i in range(len(self.mlb.classes_)):
            print("\tCategory:", self.mlb.classes_[i])
            print("\t\tPrecision:", precision[i])
            print("\t\tRecall:", recall[i])
            print("\t\tF1-score:", f1[i])

        print("\n Wrong classification:")
        count = 0
        for i in range(len(X)):
            if np.any(y_pred[i] != labels[i]):
                count += 1
                print("\tSentence:" , X[i])
                actual_label = []
                predicted_label = []
                for j in range(len(self.mlb.classes_)):
                    if labels[i][j] == 1:
                        actual_label.append(self.mlb.classes_[j])
                    if y_pred[i][j] == 1:
                        predicted_label.append(self.mlb.classes_[j])
                print("\t\tActual:", actual_label)
                print("\t\tPrediction:", predicted_label)
        print("\n Number of wrong classification:", count, "out of", len(X))

    def save_model(self, model_filename, mlb_filename):
        """
        Save trained model.

        Parameters
        ----------
        model_filename: Filename for the trained model.
        mlb_filename: Filename for the multilabel binarizer used by the trained model.

        Returns
        -------
        self
        """
        joblib.dump(self.clf, model_filename)
        joblib.dump(self.mlb, mlb_filename)
        return self

    def load_model(self, model_filename, mlb_filename):
        """
        Load trained model.

        Parameters
        ----------
        model_filename: Filename for the trained model.
        mlb_filename: Filename for the multilabel binarizer used by the trained model.

        Returns
        -------
        self
        """
        self.clf = joblib.load(model_filename)
        self.mlb = joblib.load(mlb_filename)
        return self

if __name__ == '__main__':
    X, y = AspectExtractor.read_data("../../data/aspect_extractor/train_data.csv")
    extractor = AspectExtractor()
    extractor.cross_validate(X, y, 10)
    print()
    extractor.fit(X, y)
    extractor.evaluate("../../data/aspect_extractor/test_data.csv")
    extractor.save_model("../../model/aspect_extractor.mdl", "../../model/aspect_extractor_mlb.mdl")
    extractor.load_model("../../model/aspect_extractor.mdl", "../../model/aspect_extractor_mlb.mdl")
    # print(extractor.predict(['Good location', 'Love the food here']))

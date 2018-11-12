import csv
import re
import numpy as np
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib

class SentimentExtractor(object):

    ASPECTS = ['cleanliness', 'food/drinks', 'location', 'room amenities', 'staff']
    TARGET_NAMES = ['negative', 'positive']

    def __init__(self):
        self.clf = []
        for i in range(len(self.ASPECTS)):
            pipeline = Pipeline([
                ('features', FeatureUnion(
                    transformer_list = [
                        ('bag_of_ngram', CountVectorizer(ngram_range=(1, 5)))
                    ]
                )),
                ('clf', LogisticRegression())
            ])
            self.clf.append(pipeline)

    @classmethod
    def read_data(cls, filename):
        """
        Load dataset from csv.

        Parameters
        ----------
        filename: Filename of the dataset in csv.

        Returns
        -------
        data: Review sentences grouped by aspects.
        targets: data labels grouped by aspects.
        """
        data = [[] for i in range(len(cls.ASPECTS))]
        targets = [[] for i in range(len(cls.ASPECTS))]
        regex = re.compile('[^0-9a-zA-Z]+')

        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';', quotechar='"')
            next(reader)
            for row in reader:
                text = regex.sub(' ', row[0])
                for i in range(1, len(cls.ASPECTS) + 1):
                    if row[i] != '-':
                        data[i - 1].append(text)
                        targets[i - 1].append(cls.TARGET_NAMES.index(row[i]))

        return data, targets

    def fit(self, X, y):
        """
        Train model.

        Parameters
        ----------
        X: Train data grouped by aspects.
        y: Train data labels grouped by aspects.

        Returns
        -------
        self
        """
        for i in range(len(self.ASPECTS)):
            self.clf[i].fit(X[i], y[i])
        return self

    def predict(self, X):
        """
        Predict the given data.

        Parameters
        ----------
        X: Data to predict (contain dict with sentence and aspects as key).

        Returns
        -------
        results: data and labels for the given data.
        """
        results = []
        for i in range(len(X)):
            result = {}
            result['sentence'] = X[i]['sentence']
            result['aspects'] = []
            for aspect in X[i]['aspects']:
                aspect_result = {}
                aspect_result['aspect'] = aspect
                index = self.ASPECTS.index(aspect)
                label = self.clf[index].predict([X[i]['sentence']])
                aspect_result['polarity'] = self.TARGET_NAMES[label[0]]
                result['aspects'].append(aspect_result)
            results.append(result)
        return results

    def cross_validate(self, X, y, k):
        """
        KFold cross validation.

        Parameters
        ----------
        X: Train data grouped by aspects.
        y: Train data labels grouped by aspects.
        k: Number of folds.
        """
        kf = KFold(n_splits=k)
        warnings.filterwarnings('ignore')
        print("Cross validation results:")
        for i in range(len(self.ASPECTS)):
            print("\tCategory:", self.ASPECTS[i])
            precision_scores = []
            recall_scores = []
            f1_scores = []
            for train_index, test_index in kf.split(X[i]):
                X[i] = np.array(X[i])
                y[i] = np.array(y[i])
                X_train, X_test = X[i][train_index], X[i][test_index]
                y_train, y_test = y[i][train_index], y[i][test_index]

                self.clf[i].fit(X_train, y_train)
                y_pred = self.clf[i].predict(X_test)

                precision_scores.append(precision_score(y_test, y_pred, average='macro'))
                recall_scores.append(recall_score(y_test, y_pred, average='macro'))
                f1_scores.append(f1_score(y_test, y_pred, average='macro'))

            print("\t\tPrecision:", np.array(precision_scores).mean())
            print("\t\tRecall:", np.array(recall_scores).mean())
            print("\t\tF1-score:", np.array(f1_scores).mean())

    def evaluate(self, test_data_filename):
        """
        Evaluate the trained model using test data.

        Parameters
        ----------
        test_data_filename: Filename for the test data.
        """
        X, y = SentimentExtractor.read_data(test_data_filename)

        print("Evaluation results:")
        for i in range(len(self.ASPECTS)):
            print("\tCategory:", self.ASPECTS[i])
            y_pred = self.clf[i].predict(X[i])

            print("\t\tPrecision:", precision_score(y[i], y_pred, average='macro'))
            print("\t\tRecall:", recall_score(y[i], y_pred, average='macro'))
            print("\t\tF1-score:", f1_score(y[i], y_pred, average='macro'))

            print("\t\tWrong classification:")
            count = 0
            for j in range(len(X[i])):
                if y_pred[j] != y[i][j]:
                    count += 1
                    print("\t\t\tSentence:" , X[i][j])
                    print("\t\t\tActual:", y[i][j])
                    print("\t\t\tPrediction:", y_pred[j])
            print("\t\tNumber of wrong classification:", count, "out of", len(X[i]))

    def save_model(self, model_filename):
        """
        Save trained model.

        Parameters
        ----------
        model_filename: Filename for the trained model.

        Returns
        -------
        self
        """
        for i in range(len(self.ASPECTS)):
            joblib.dump(self.clf[i], model_filename + str(i))
        return self

    def load_model(self, model_filename):
        """
        Load trained model.

        Parameters
        ----------
        model_filename: Filename for the trained model.

        Returns
        -------
        self
        """
        for i in range(len(self.ASPECTS)):
            self.clf[i] = joblib.load(model_filename + str(i))
        return self

if __name__ == '__main__':
    X, y = SentimentExtractor.read_data("../../data/sentiment_extractor/train_data.csv")
    extractor = SentimentExtractor()
    extractor.cross_validate(X, y, 10)
    print()
    extractor.fit(X, y)
    extractor.evaluate("../../data/sentiment_extractor/test_data.csv")
    extractor.save_model("../../model/sentiment_extractor.mdl")
    extractor.load_model("../../model/sentiment_extractor.mdl")
    # print(extractor.predict([{'sentence': 'Good location', 'aspects': ['location']}, {'sentence': 'Love the food here', 'aspects': ['food/drinks']}]))

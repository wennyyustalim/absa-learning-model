# from sentiment_feature_extractor import SentimentFeatureExtractor
from item_selector import ItemSelector
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, \
    accuracy_score
from sklearn.externals import joblib
from nltk.util import skipgrams
from nltk.corpus import stopwords
from collections import OrderedDict
import numpy as np
import csv
import re
import functools
import os


class SentimentExtractor:
    def __init__(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

        # stopword_filename = os.path.join(
        #     project_path, "preprocess/resource/stopword.txt")

        # stopWords = set(stopwords.words('english'))

        # with open(stopword_filename, "r") as f:
        #     stopword = f.readlines()
        # self.stopword = [x.rstrip() for x in stopword]

        self.categories = ['staff',
                           'amenities',
                           'food/drinks',
                           'cleanliness',
                           'location']
        # self.target_names = ['positive', 'negative', 'neutral']
        self.target_names = ['1', '0']
        self.train_data = []
        self.train_target = []
        train_data, train_target, _ = self.read_data(
            os.path.join(project_path, "../csv_out/dataset-absa-polarity.csv"))

        # print(len(self.train_data))

        for i in range(len(self.categories)):
            self.train_data.append(np.array(train_data[i]))
            self.train_target.append(np.array(train_target[i]))

        self.model_filenames = [
            os.path.join("./model_out/sentiment_staff.model"),
            os.path.join("./model_out/sentiment_amenities.model"),
            os.path.join("./model_out/sentiment_food_drinks.model"),
            os.path.join("./model_out/sentiment_cleanliness.model"),
            os.path.join("./model_out/sentiment_location.model")
        ]

    def read_data(self, filename):
        all_data = []
        data = [OrderedDict(), OrderedDict(), OrderedDict(),
                OrderedDict(), OrderedDict()]
        targets = [[], [], [], [], []]
        regex = re.compile('[^0-9a-zA-Z]+')
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            next(reader)
            j = 0
            for row in reader:
                text = regex.sub(' ', row[0])
                # text = " ".join(x for x in text.split() if x not in self.stopword)

                all_data.append(regex.sub(' ', text))
                for i in range(1, len(self.categories) + 1):
                    if row[i] != "-1":
                        data[i - 1][j] = regex.sub(' ', text)
                        targets[i - 1].append(self.target_names.index(row[i]))
                j += 1

        return data, targets, all_data

    def get_pipeline(self, index):
        return Pipeline([
            # ('data', SentimentFeatureExtractor()),

            ('features', FeatureUnion(
                transformer_list=[
                    ('bag_of_ngram', Pipeline([
                        # ('selector', ItemSelector(key='text')),
                        ('ngram', CountVectorizer(ngram_range=(1, 5))),
                    ])),
                ]
            )),

            ('clf', LogisticRegression())
        ])

    def train(self):
        for index in range(len(self.categories)):
            # print(list(self.train_data[index].values()))
            # print(np.array(list(self.train_data[index].values()))[0])
            train_data = np.array(list(self.train_data[index].values()))
            # pipeline = self.get_pipeline(index)
            feature_extractor = CountVectorizer(ngram_range=(1, 2))
            # print(train_data[0], train_data[1])
            feature = feature_extractor.fit_transform(train_data)
            model = LogisticRegression()
            model.fit(feature, self.train_target[index])
            # model = pipeline.fit(train_data, self.train_target[index])
            joblib.dump(
                feature_extractor, './model_out/feature-extractor-' + str(index) + '.model')
            joblib.dump(model, self.model_filenames[index])

    def evaluate_cross_validation(self):
        n = 10

        for index in range(len(self.train_data)):
            X_folds = np.array_split(self.train_data[index].values(), n)
            y_folds = np.array_split(self.train_target[index], n)

            precision_scores = []
            recall_scores = []
            f1_scores = []

            pipeline = self.get_pipeline(index)

            print("CATEGORY: " + self.categories[index])
            for k in range(n):
                X_train = list(X_folds)
                X_test = X_train.pop(k)
                X_train = np.concatenate(X_train)
                y_train = list(y_folds)
                y_test = y_train.pop(k)
                y_train = np.concatenate(y_train)

                model = pipeline.fit(X_train, y_train)
                predicted = model.predict(X_test)

                # print classification_report(y_test, predicted)
                # print confusion_matrix(y_test, predicted)

                precision_scores.append(precision_score(
                    y_test, predicted, average=None).mean())
                recall_scores.append(recall_score(
                    y_test, predicted, average=None).mean())
                f1_scores.append(
                    f1_score(y_test, predicted, average=None).mean())

            print("\tPrecision: ", np.array(precision_scores).mean())
            print("\tRecall: ", np.array(recall_scores).mean())
            print("\tF1-score: ", np.array(f1_scores).mean())

    def evaluate(self, test_filename):
        test_data, test_target, all_data = self.read_data(test_filename)

        for i in range(len(self.categories)):
            # print(test_data[i].values())
            model = joblib.load(self.model_filenames[i])
            feature_extractor = joblib.load(
                './model_out/feature-extractor-' + str(i) + '.model')
            train_data = np.array(list(test_data[i].values()))
            # pipeline = self.get_pipeline(index)
            # print(train_data[0], train_data[1])
            feature = feature_extractor.transform(train_data)
            predicted = model.predict(feature)

            for key, actual, predict in zip(test_data[i], test_target[i], predicted):
                if actual != predict:
                    print(key+1, " ", test_data[i][key])
                    print("\tactual: ", actual)
                    print("\tpredict: ", predict)

            print("CATEGORY: " + self.categories[i])
            print("\tPrecision: ", np.array(precision_score(
                test_target[i], predicted, average=None)).mean())
            print("\tRecall: ", np.array(recall_score(
                test_target[i], predicted, average=None)).mean())
            print("\tF1-score: ",
                  np.array(f1_score(test_target[i], predicted, average=None)).mean())

            # print "CATEGORY: " + self.categories[i]
            # print "\tPrecision: ", np.array(precision_score(test_target[i], predicted, average=None))
            # print "\tRecall: ", np.array(recall_score(test_target[i], predicted, average=None))
            # print "\tF1-score: ", np.array(f1_score(test_target[i], predicted, average=None))

            # print classification_report(test_target[i], predicted)
            # print confusion_matrix(test_target[i], predicted)

    def evaluate_accumulative(self, actual_data_filename, test_filename):
        actual_data, actual_target, all_data = self.read_data(
            actual_data_filename)
        test_data, test_target, all_data = self.read_data(test_filename)

        for i in range(len(self.categories)):
            print("CATEGORY: " + self.categories[i])
            model = joblib.load(self.model_filenames[i])
            predicted = model.predict(test_data[i].values())

            test_data[i] = OrderedDict(test_data[i])
            class_precision = []
            class_recall = []
            class_f1 = []
            for j in range(len(self.target_names)):
                correct = 0.0
                incorrect = 0.0
                size_class_prediction = 0
                size_class_actual = 0

                for target in actual_target[i]:
                    if target == j:
                        size_class_actual += 1

                for predict in predicted:
                    if predict == j:
                        size_class_prediction += 1

                for key, target in zip(actual_data[i], actual_target[i]):
                    if key in test_data[i]:
                        prediction = predicted[test_data[i].keys().index(key)]
                        if prediction == j:
                            if target == prediction:
                                correct += 1

                precision = 0
                recall = 0
                f1 = 0

                if size_class_prediction > 0:
                    precision = correct / size_class_prediction

                if size_class_actual > 0:
                    recall = correct / size_class_actual

                if recall > 0:
                    f1 = (2 * precision * recall) / (precision + recall)

                class_precision.append(precision)
                class_recall.append(recall)
                class_f1.append(f1)

            # print "\tPrecision: ", class_precision
            # print "\tRecall: ", class_recall
            # print "\tF1-score: ", class_f1

            print("\tPrecision: ", np.array(class_precision).mean())
            print("\tRecall: ", np.array(class_recall).mean())
            print("\tF1-score: ", np.array(class_f1).mean())

    def predict(self, category, test_data):
        results = []
        if category == "staff":
            model = joblib.load(self.model_filenames[0])
        elif category == "amenities":
            model = joblib.load(self.model_filenames[1])
        elif category == "food/drinks":
            model = joblib.load(self.model_filenames[2])
        elif category == "cleanliness":
            model = joblib.load(self.model_filenames[3])
        elif category == "location":
            model = joblib.load(self.model_filenames[4])

        predicted = model.predict(np.array(test_data))
        for j in range(len(predicted)):
            results.append(self.target_names[predicted[j]])

        return results


if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

    sentiment_extractor = SentimentExtractor()
    sentiment_extractor.train()
    # sentiment_extractor.evaluate_cross_validation()

    sentiment_extractor.evaluate(project_path +
                                 "/aspect-based-sentiment-analysis-scraping/csv_out/dataset-absa-polarity.test.csv")
    # sentiment_extractor.evaluate_accumulative("../../data/sentiment_extraction/test_data_3.csv",
    #                                           "../../data/sentiment_extraction/test_data_cumulative.csv")

import json
import argparse
from classifier import NaiveBayesDocumentClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

"""
A document classifier using Bernoulli Naive Bayes from sklearn library.
"""


class SklearnBernoulli(NaiveBayesDocumentClassifier):


    def __init__(self):
        super().__init__()


    def train(self, features, labels):
        # Load your training data
        with open("train.json") as file:
            data = json.load(file)
        vocab = data['vocabulary']

        texts = []
        for doc in features:
            text = ' '.join(features[doc].keys())
            texts.append(text)

        vectorizer = CountVectorizer(vocabulary=vocab, binary=True)
        X = vectorizer.transform(texts)
        y = list(labels.values())

        self.model = BernoulliNB(alpha=self.EPSILON * 200, fit_prior=True)
        self.model.fit(X, y)
        self.save_model()

    def apply(self, features):
        # Load your training data
        with open("train.json") as file:
            data = json.load(file)
        vocab = data['vocabulary']

        if self.model is None:
            self.load_model()

        texts = []
        for doc in features:
            text = ' '.join(features[doc].keys())
            texts.append(text)

        vectorizer = CountVectorizer(vocabulary=vocab, binary=True)
        X = vectorizer.transform(texts)

        y = self.model.predict(X)

        # Map the document names to the predicted labels
        prediction = {doc: label for doc, label in zip(features.keys(), y)}

        """
        # Print the predicted labels
        for doc, c in prediction.items():
            print(doc, ':', c)
        """

        return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A document classifier using Bernoulli Naive Bayes from sklearn library.')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--apply', help="apply the classifier (you'll need to train or load"\
                                        "a trained model first)", action='store_true')
    args = parser.parse_args()

    classifier = SklearnBernoulli()

    def read_json(path):
        with open(path) as f:
            data = json.load(f)['docs']
            features,labels = {},{}
            for f in data:
                features[f] = data[f]['tokens']
                labels[f] = data[f]['label']
        return features,labels
    
    if args.train:
        features,labels = read_json('train.json')
        classifier.train(features, labels)

    if args.apply:
        features,labels = read_json('test.json')
        result = classifier.apply(features)

        # Measure error rate
        false_predictions = 0

        for doc in result:
            if result[doc] != labels[doc]:
                false_predictions += 1

        error_rate = (false_predictions / len(result)) * 100
        print('Error:', error_rate, '%')

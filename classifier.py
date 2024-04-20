import argparse, json, pickle, numpy as np

"""
A document classifier using boolean bag-of-words features and Naive Bayes.
I implemented both Naive Bayes and the boolean bag-of-word features from scratch.
Additionally, I implemented the classifier using the sklearn library for comparison. (see classifier_sklearn.py)
"""


class NaiveBayesDocumentClassifier:

    
    def __init__(self):
        self.model = None
        # Set epsilon smoothing factor to avoid zero probabilities
        self.EPSILON = 0.0007


    def load_model(self):
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)


    def save_model(self):
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)


    def train(self, features, labels):
        """
        Trains the document classifier and stores the model in 'self.model'.

        @type features: dict
        @param features: Each entry in 'features' represents a document by its bag-of-words vector. 
                         For each document in the dataset, 'features' contains all terms occurring
                         in the document and their frequency in the document:
                         {
                           'doc1.html':
                              {
                                'the' : 7,   # 'the' occurs seven times
                                'world': 3, 
                                ...
                              },
                           'doc2.html':
                              {
                                'community' : 2,
                                'college': 1, 
                                ...
                              },
                            ...
                         }
        @type labels: dict
        @param labels: 'labels' contains the class labels for all documents
                       in dictionary form:
                       {
                           'doc1.html': 'arts',       # doc1.html belongs to class 'arts'
                           'doc2.html': 'business',
                           'doc3.html': 'sports',
                           ...
                       }
        """

        # Load your training data
        with open("train.json") as file:
          data = json.load(file)
        vocab = data['vocabulary']


        # Create boolean bag-of-words features
        feat_dict = {}

        for doc,words in features.items():
            feat_dict[doc] = {}
            for voc_word in vocab:
                if voc_word in words:
                    feat_dict[doc][voc_word] = 1
                else:
                    feat_dict[doc][voc_word] = 0


        # Get class probabilities, [prior P(y)]
        classes = set(labels.values())
        class_probabilities = {}

        for c in classes:
            for doc in labels:
                if labels[doc] == c:
                    if c in class_probabilities:
                        class_probabilities[c] += 1
                    else:
                        class_probabilities[c] = 1
            class_probabilities[c] /= len(labels)
        

        # Calculate the word probabilities, [class conditional density (ccd)  P(xi|y)]
        word_probabilities = {}

        for c in classes:
            word_probabilities[c] = {}
            for word in vocab:
                word_probabilities[c][word] = 0
                for doc in feat_dict:
                    if labels[doc] == c:
                        if feat_dict[doc][word] == 1:
                            word_probabilities[c][word] += 1
                # Divide by the total number of documents in the class
                # P(xi|y) = #(xi,y) / #(y)
                word_probabilities[c][word] /= (class_probabilities[c] * len(labels))

        self.model = {'prior': class_probabilities, 'ccd': word_probabilities}
        self.save_model()

        """
        # Print the word probabilities of the top 20 words for each class
        for c in classes:
            print('Class: ', c)
            sorted_word_probabilities = sorted(word_probabilities[c].items(), key=lambda x: x[1], reverse=True)
            for i in range(20):
                print(sorted_word_probabilities[i][0], ':', sorted_word_probabilities[i][1])
            print()
        """

        
    def apply(self, features):
        """
        Applies the classifier to a set of documents. Requires the classifier
        to be trained (i.e., you need to call train() before you can call apply()).

        @type features: dict
        @param features: see above (documentation of train())

        @rtype: dict
        @return: For each document in 'features', apply() returns the estimated class.
                 The return value is a dictionary of the form:
                 {
                   'doc1.html': 'arts',
                   'doc2.html': 'travel',
                   'doc3.html': 'sports',
                   ...
                 }
        """
        # Load your training data
        with open("train.json") as file:
          data = json.load(file)
        vocab = data['vocabulary']

        if self.model == None:
            self.load_model()


        # Smooth the word probabilities with epsilon before applying the classifier
        for c,words in self.model['ccd'].items(): # type: ignore
            for word in words:
                if words[word] < self.EPSILON:
                    words[word] = self.EPSILON
                elif words[word] > 1 - self.EPSILON:
                    words[word] = 1 - self.EPSILON


        # Apply the classifier (Note that we are estimating using log likelihood)
        posterior_probabilities = {}

        for doc,words in features.items():
            posterior_probabilities[doc] = {}
            for c in self.model['prior']: # type: ignore
                # Initialize with the class probability [log(P(y))]
                posterior_probabilities[doc][c] = np.log(self.model['prior'][c]) # type: ignore
                for word in vocab:
                    # Add ccd [log(P(xi|y)) or log(1-P(xi|y))] for each feature
                    if word in words:
                        posterior_probabilities[doc][c] += np.log(self.model['ccd'][c][word]) # type: ignore
                    else:
                        posterior_probabilities[doc][c] += np.log((1 - self.model['ccd'][c][word])) # type: ignore


        # Get the class with the maximum posterior probability for each document
        max_posterior = {}

        for doc in posterior_probabilities:
            for c in posterior_probabilities[doc]:
                if doc in max_posterior:
                    if posterior_probabilities[doc][c] > max_posterior[doc][1]:
                        max_posterior[doc] = (c, posterior_probabilities[doc][c])
                else:
                    max_posterior[doc] = (c, posterior_probabilities[doc][c])

        # Get the class names for the predicted classes
        prediction = { doc: max_posterior[doc][0] for doc in max_posterior } 

        """
        # Print the predicted classes
        for doc, c in prediction.items():
            print(doc, ':', c)
        """

        return prediction  

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A document classifier using boolean bag-of-words features and Naive Bayes.')
    parser.add_argument('--train', help="Train the classifier", action='store_true')
    parser.add_argument('--apply', help="Apply the classifier (you'll need to train or load"\
                                        "a trained model first)", action='store_true')
    args = parser.parse_args()

    classifier = NaiveBayesDocumentClassifier()

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

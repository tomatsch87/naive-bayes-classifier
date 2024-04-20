# Naive Bayes Document Classifier from Scratch

Implementing a Naive Bayes document classifier from scratch in Python. The project also includes an implementation using the Bernoulli Naive Bayes classifier from the sklearn library for comparison.

## Files

- `classifier.py`: This file contains the implementation of a Naive Bayes document classifier from scratch. The classifier is trained on a set of documents and their labels, and can then be applied to classify a new set of documents.

- `classifier_sklearn.py`: This file contains an implementation of a document classifier using the Bernoulli Naive Bayes classifier from the sklearn library. It inherits from the Naive Bayes document classifier implemented in `classifier.py`.

## Usage

To train the classifier, run the `classifier.py` script with the `--train` argument. This will train the classifier on the data in `train.json`.

```bash
python classifier.py --train
```

To apply the classifier, run the `classifier.py` script with the `--apply` argument. This will apply the classifier to the data in `test.json` and print the error rate.

```bash
python classifier.py --apply
```

## Data Format

The training and test data should be in JSON format. Each document is represented by a dictionary with the document's tokens and their frequencies, and the document's label.
Also the vocabulary of all the documents is to be included in the JSON file as a separate dictionary with the tokens and their frequencies.

For example:

```json
{
  "docs": {
    "doc1.html": {
      "tokens": {
        "the": 7,
        "world": 3,
        ...
      },
      "label": "arts"
    },
    "doc2.html": {
      "tokens": {
        "community": 2,
        "college": 1,
        ...
      },
      "label": "business"
    },
    ...
    "vocabulary": {
        "the": 10,
        "world": 5,
        ...
        },
    }
  }
}
```

## Dependencies

This project requires Python 3 and the following Python libraries installed:

- [sklearn](https://scikit-learn.org/stable/)
- [numpy](https://numpy.org/)
- [json](https://docs.python.org/3/library/json.html)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [pickle](https://docs.python.org/3/library/pickle.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

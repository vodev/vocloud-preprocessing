"""This module contains a GroupedClassifier class"""
import sklearn.base as base
class GroupedClassifier(base.BaseEstimator):
    """GroupedClassifier is meant to group together classifiers
       that should run be fitted to the same data. It is meant
       to make scoring of many classifiers easier"""
    def __init__(self, classifiers=None):
        super(GroupedClassifier, self).__init__()
        if classifiers is None:
            self.classifiers = []
        else:
            self.classifiers = classifiers

    def add_classifier(self, classifier):
        '''Adds a classifier to the group.
        The classifier must be fitted to the same data
        as the others, or fit method must be run afterwards
        to fit all the classifiers to the same data'''
        self.classifiers.append(classifier)

    def clear(self):
        '''Clears classifiers'''
        self.classifiers.clear()

    def fit(self, samples, labels, sample_weigth):
        '''Fits all the classifiers with the same data'''
        for classifier in self.classifiers():
            classifier.fit(samples, labels, sample_weigth)

    def score_all(self, samples, labels, sample_weigth):
        '''Runs a scoring method on all the classifiers in the group and returns
           a list of tuples (classifier, classifier_score)'''
        scores = []
        for classifier in self.classifiers:
            scores.append((classifier, classifier.score(samples, labels, sample_weigth)))
        return scores

    def predict_all(self, samples):
        """Runs a prediction on all the classifiers in the group
           with given samples and returns
           a list of tuples(classifier, classifier_score)"""
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier, classifier.predict(samples))
        return predictions
"""This module contains a GroupedClassifier class"""
import sklearn.metrics as metrics
import sklearn.cross_validation as cv
import matplotlib.pyplot as plt
from sklearn.metrics.metrics import UndefinedMetricWarning
import numpy as np

import base.grouped_estimator as grouped_estimator


def __plot_matrix(matrix, title):
    fig = plt.figure()
    plt.matshow(matrix, fignum=0)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(4), range(1, 5))
    plt.yticks(range(4), range(1, 5))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_matrices(matrices):
    figs = []
    for matrix in matrices:
        normalized_mat = normalize_matrix(matrix[1])
        figs.append(__plot_matrix(normalized_mat, matrix[0]))
    return figs


def normalize_matrix(matrix):
    new_matrix = []
    for row in matrix:
        total_sum = sum(row)
        if total_sum == 0:
            total_sum = 1
        new_row = []
        for num in row:
            new_row.append(num / total_sum)
        new_matrix.append(new_row)
    return np.matrix(new_matrix)


def plot_scores(results, labels=None):
    fig = plt.figure()
    if labels is None:
        labels = [result[0] for result in results]
    scores = [result[1] for result in results]
    x_axis = range(1, len(results) + 1)
    plt.bar(x_axis, scores)
    plt.xticks(x_axis, labels)
    plt.tick_params(axis='x', labelsize=8)
    plt.ylim(ymin=0, ymax=1)
    return fig


def plot_crossvalidation(crossvalidation_score, labels=None):
    fig = plt.figure()
    if labels is None:
        labels = [str(score[0]) for score in crossvalidation_score]
    means = [score[1].mean() for score in crossvalidation_score]
    stds = [score[1].std() for score in crossvalidation_score]
    x_axis = range(1, len(means) + 1)
    plt.xlim(xmin=0, xmax=len(crossvalidation_score) + 1)
    plt.ylim(ymin=0, ymax=1)
    plt.errorbar(x_axis, means, stds, marker='^', linestyle='none', label='Crossvalidation', rasterized=False)
    plt.xticks(x_axis, labels)
    plt.figsize = (20, 2)
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=8)
    return fig


class GroupedClassifier(grouped_estimator.GroupedEstimator):
    """GroupedClassifier is meant to group together classifiers
       that should run be fitted to the same data. It is meant
       to make scoring of many classifiers easier"""

    def __init__(self, classifiers=None, labels=None):
        super(GroupedClassifier, self).__init__(classifiers, labels)

    def fit(self, samples, labels, weights=None):
        '''Fits all the classifiers with the same data'''
        for classifier in self.estimators:
            try:
                classifier.fit(samples, labels, weights=weights)
            except TypeError:
                classifier.fit(samples, labels)

    def score_all(self, samples, labels, weights=None):
        '''Runs a scoring method on all the classifiers in
           the group and returns a list of tuples
           (classifier, classifier_score)'''
        scores = []
        for classifier_label, classifier in self.estimators.items():
            predictions = classifier.predict(samples)
            try:
                score = metrics.f1_score(labels, predictions, average='macro', sample_weight=weights)
            except UndefinedMetricWarning:
                print('Classifier ' + classifier_label + ' has zero f1')
            scores.append((classifier_label, score))
        return scores

    def predict_all(self, samples):
        """Runs a prediction on all the classifiers in the group
           with given samples and returns
           a list of tuples(classifier, classifier_score)"""
        predictions = []
        for classifier_label, classifier in self.estimators.items():
            predictions.append((classifier_label, classifier.predict(samples)))
        return predictions

    def confusion_matrix(self, samples, labels):
        predictions = self.predict_all(samples)
        confusion_matrices = []
        for prediction in predictions:
            result = (prediction[0], metrics.confusion_matrix(prediction[1],
                                                              labels).transpose())
            confusion_matrices.append(result)
        return confusion_matrices

    def crossvalidation_score(self, samples, labels, k=5):
        scores = []
        for classifier_label, classifier in self.estimators.items():
            try:
                score = cv.cross_val_score(classifier, samples,
                                           labels,
                                           n_jobs=-1,
                                           cv=k,
                                           scoring='f1')
            except UndefinedMetricWarning:
                print('Classifier ' + classifier_label + ' has zero f1')
            scores.append((classifier_label, score))
        return scores


    def feature_importances(self):
        importances = []
        for classifier_label, classifier in self.estimators.items():
            try:
                importances.append((classifier_label, classifier.feature_importances_))
            except AttributeError:
                pass
        return importances

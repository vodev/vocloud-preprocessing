"""This module contains a GroupedClassifier class"""
import sklearn.metrics as metrics
import sklearn.cross_validation as cv
import matplotlib.pyplot as plt
import base.grouped_estimator as grouped_estimator

def __plot_matrix(matrix, title):
    plt.figure()
    plt.matshow(matrix)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_matrices(matrices):
    for matrix in matrices:
        __plot_matrix(matrix[1], matrix[0])
    plt.show()
    plt.close()

def plot_scores(results, labels=None):
    plt.figure()
    if labels is None:
        labels = [result[0] for result in results]
    scores = [result[1] for result in results]
    x_axis = range(1, len(results) + 1)
    plt.bar(x_axis, scores)
    plt.xticks(x_axis, labels)
    plt.ylim(ymin=0, ymax=1)

def plot_crossvalidation(crossvalidation_score, labels=None):
    plt.figure()
    if labels is None:
        labels = [str(score[0]) for score in crossvalidation_score]
    means = [score[1].mean() for score in crossvalidation_score]
    stds = [score[1].std() for score in crossvalidation_score]
    x_axis = range(1, len(means) + 1)
    plt.xlim(xmin=0, xmax=len(crossvalidation_score) + 1)
    plt.ylim(ymin=0, ymax=1)
    plt.errorbar(x_axis, means, stds, marker='^', linestyle='none', label='Crossvalidation', rasterized=False)
    plt.xticks(x_axis, labels)
    plt.figsize=(20, 2)
    plt.tight_layout()
    plt.show()

class GroupedClassifier(grouped_estimator.GroupedEstimator):
    """GroupedClassifier is meant to group together classifiers
       that should run be fitted to the same data. It is meant
       to make scoring of many classifiers easier"""
    def __init__(self, classifiers=None):
        super(GroupedClassifier, self).__init__(classifiers)

    def fit(self, samples, labels):
        '''Fits all the classifiers with the same data'''
        for classifier in self.estimators:
            classifier.fit(samples, labels)

    def score_all(self, samples, labels):
        '''Runs a scoring method on all the classifiers in
           the group and returns a list of tuples
           (classifier, classifier_score)'''
        scores = []
        for classifier in self.estimators:
            scores.append((classifier, classifier.score(samples, labels)))
        return scores

    def predict_all(self, samples):
        """Runs a prediction on all the classifiers in the group
           with given samples and returns
           a list of tuples(classifier, classifier_score)"""
        predictions = []
        for classifier in self.estimators:
            predictions.append((classifier, classifier.predict(samples)))
        return predictions

    def confusion_matrix(self, samples, labels):
        predictions = self.predict_all(samples)
        confusion_matrices = []
        for prediction in predictions:
            result = (prediction[0], metrics.confusion_matrix(prediction[1],
                                                              labels))
            confusion_matrices.append(result)
        return confusion_matrices

    def crossvalidation_score(self, samples, labels, k=5):
        scores = []
        for classifier in self.estimators:
            score = cv.cross_val_score(classifier, samples,
                                       labels, n_jobs=-1, cv=k)
            scores.append((classifier, score))
        return scores

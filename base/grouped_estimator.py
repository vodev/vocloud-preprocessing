import sklearn.base as base

class GroupedEstimator(base.BaseEstimator):
    """GroupedClassifier is meant to group together classifiers
       that should run be fitted to the same data. It is meant
       to make scoring of many classifiers easier"""
    def __init__(self, classifiers=None):
        super(GroupedEstimator, self).__init__()
        if classifiers is None:
            self.estimators = []
        else:
            self.estimators = classifiers

    def add_estimator(self, classifier):
        '''Adds a classifier to the group.
        The classifier must be fitted to the same data
        as the others, or fit method must be run afterwards
        to fit all the classifiers to the same data'''
        self.estimators.append(classifier)

    def clear(self):
        '''Clears classifiers'''
        self.estimators.clear()

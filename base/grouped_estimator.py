import sklearn.base as base


class GroupedEstimator(base.BaseEstimator):
    """GroupedClassifier is meant to group together classifiers
       that should run be fitted to the same data. It is meant
       to make scoring of many classifiers easier"""

    def __init__(self, estimators=None, labels=None, group_name=None):
        super(GroupedEstimator, self).__init__()
        if labels is None:
            self.labels = self.__generate_labels(estimators)
        elif len(labels) == len(estimators):
            self.labels = labels
        else:
            raise ValueError('The length of estimators and labels must be the same')
        self.estimators = {}
        for idx, label in enumerate(self.labels):
            self.estimators[label] = estimators[idx]
        if group_name is None:
            self.group_name = 'Group'

    @staticmethod
    def __generate_labels(estimators):
        return ['estimator ' + str(i) for i in range(len(estimators))]

    def add_estimator(self, estimator, label=None):
        '''Adds a classifier to the group.
        The classifier must be fitted to the same data
        as the others, or fit method must be run afterwards
        to fit all the classifiers to the same data'''
        if label is None:
            label = 'estimator ' + str(len(self.estimators))
        self.estimators[label] = estimator

    def clear(self):
        '''Clears classifiers'''
        self.estimators.clear()

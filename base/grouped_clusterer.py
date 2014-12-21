import sklearn.metrics as metrics
import base.grouped_estimator as grouped_estimator

def save_clustering_results(data, results):
    for result in results:
        print(result[0])
        for idx, index in enumerate(data.index):
            print('%s of type %s in cluster %s' %(index, data.loc[index]['class'], result[1][idx]))

class GroupedClusterer(grouped_estimator.GroupedEstimator):
    """GroupedClassifier is meant to group together classifiers
       that should run be fitted to the same data. It is meant
       to make scoring of many classifiers easier"""
    def __init__(self, clusterers=None):
        super(GroupedClusterer, self).__init__(clusterers)

    def fit(self, samples):
        '''Fits clusters for samples'''
        for estimator in self.estimators:
            estimator.fit(samples)

    def fit_predict(self, samples):
        predictions = []
        for estimator in self.estimators:
            prediction = estimator.fit_predict(samples)
            predictions.append((estimator, prediction))
        return predictions

    def fit_transform(self, samples):
        transformations = []
        for estimator in self.estimators:
            transformation = estimator.fit_transform(samples)
            transformations.append((estimator, transformation))
        return transformations

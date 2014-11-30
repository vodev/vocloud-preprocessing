'''Will provide models for data and test their accuracies'''
import sklearn.ensemble as ensemble
import sklearn.cluster as clustering
import sklearn.metrics as metrics
from sklearn import svm
def fit_classifier_model(data_frame, cls, data_columns=None,
                         class_columns=None,
                         sample_weigths=None, **parameters):
    '''Fits a random forest'''
    if(data_columns is None):
        
    forest = cls(parameters)
    data = data_frame.as_matrix[data_columns]
    class_column = data_frame['class'].as_matrix()
    forest = forest.fit(data, class_column, sample_weigths)
    return forest

def fit_clustering_model(data_frame, cls, data_columns=None,**parameters):
    '''Something'''
from sklearn.preprocessing._weights import balance_weights
import sklearn.grid_search as gs
import sklearn.cross_validation as cv
'''Will provide models for data and test their accuracies'''


def fit_classifier_model(data_frame, cls, data_columns=None,
                         class_columns='class', **kwargs):
    '''Fits a classifier model specified by cls'''
    if data_columns is None:
        data_columns = data_frame.columns[0:-2]
    if class_columns is None:
        class_columns = data_frame.columns[-1:]
    classifier = cls(**kwargs)
    data = data_frame.as_matrix(data_columns)
    class_column = data_frame[class_columns].as_matrix().ravel()
    weights = balance_weights(class_column)
    try:
        classifier = classifier.fit(data, class_column, sample_weights=weights)
    except TypeError:
        classifier = classifier.fit(data, class_column)
    return classifier


def find_best_params(data_frame, estimator, parameter_dict,
                     data_columns=None, class_columns='class',
                     **kwargs):
    clf = fit_classifier_model(data_frame, estimator, data_columns=data_columns,
                               class_columns=class_columns, **kwargs)
    if data_columns is None:
        data_columns = data_frame.columns[0:-2]
    if class_columns is None:
        class_columns = data_frame.columns[-1:]
    data = data_frame.as_matrix(data_columns)
    class_column = data_frame[class_columns].as_matrix().ravel()
    parameter_search = gs.GridSearchCV(clf, parameter_dict, scoring='f1',
                                       iid=False,
                                       cv=cv.StratifiedKFold(class_column, n_folds=3))
    parameter_search.fit(data, class_column)
    print(parameter_search.best_params_)
    return parameter_search.best_estimator_


def fit_clustering_model(data_frame, cls, data_columns=None, **kwargs):
    '''Fits a classifier model'''
    if data_columns is None:
        data_columns = data_frame.columns[0:-2]

    model = cls(**kwargs)
    data = data_frame.as_matrix(data_columns)
    model = model.fit(data)
    return model
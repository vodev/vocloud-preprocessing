'''Will provide models for data and test their accuracies'''
def fit_classifier_model(data_frame, cls, data_columns=None,
                         class_columns='class', **parameters):
    '''Fits a classifier model specified by cls'''
    if data_columns is None:
        data_columns = data_frame.columns[0:-2]
    if class_columns is None:
        class_columns = data_frame.columns[-1:]
    classifier = cls(**parameters)
    data = data_frame.as_matrix(data_columns)
    class_column = data_frame[class_columns].as_matrix().ravel()
    classifier = classifier.fit(data, class_column)
    return classifier

def fit_clustering_model(data_frame, cls, data_columns=None, **parameters):
    '''Fits a classifier model'''
    if data_columns is None:
        data_columns = data_frame.columns[0:-2]

    model = cls(**parameters)
    data = data_frame.as_matrix(data_columns)
    model = model.fit(data)
    return model
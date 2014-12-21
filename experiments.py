import base.data_handler as dh
import base.grouped_classifier as grouped_classifier
import base.grouped_clusterer as grouped_clusterer
import base.models as models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


def plot_results(results, labels=None):
    matrices = results['matrix']
    score = results['score']
    crossvalidation = results['crossvalidation']
    grouped_classifier.plot_matrices(matrices)
    grouped_classifier.plot_scores(score, labels)
    grouped_classifier.plot_crossvalidation(crossvalidation, labels)


def run_pca_tests(spectra_df, n_components=5):
    '''Runs tests using PCA'''
    kinds = ['ICA', 'PCA', 'Kernel']
    results = {}
    for kind in kinds:
        decomposed = dh.decompose(spectra_df, data_cols=spectra_df.columns[0:-2], n_components=n_components, kind=kind)
        transformed_train_set, transformed_test_set = dh.split_train_set(decomposed)
        classifiers = train_classifiers(transformed_train_set)
        test_set_data = transformed_test_set.as_matrix(decomposed.columns[0:-2])
        test_set_labels = transformed_test_set['class'].as_matrix().ravel()
        score = classifiers.score_all(test_set_data, test_set_labels)
        results['classifiers'] = classifiers
        results['score'] = score
        results['matrix'] = classifiers.confusion_matrix(test_set_data, test_set_labels)
        results['crossvalidation'] = run_crossvalidation_on_group(classifiers, transformed_train_set)
    return results


def run_without_pca(spectra_df):
    results = {}
    train_set, test_set = dh.split_train_set(spectra_df)
    classifiers = train_classifiers(train_set)
    test_set_data = test_set.as_matrix(test_set.columns[0:-2])
    test_set_labels = test_set['class'].as_matrix().ravel()
    results['classifiers'] = classifiers
    results['score'] = classifiers.score_all(test_set_data, test_set_labels)
    results['matrix'] = classifiers.confusion_matrix(test_set_data, test_set_labels)
    results['crossvalidation'] = run_crossvalidation_on_group(classifiers,
                                                              train_set)
    return results


def run_crossvalidation_on_group(group, validation_set):
    data_columns = validation_set.columns[:-2]
    class_col = 'class'
    scores = group.crossvalidation_score(validation_set.as_matrix(data_columns),
                                         validation_set[class_col].as_matrix().ravel())
    return scores

def train_classifiers(train_set):
    data_columns = train_set.columns[:-2]
    class_col = 'class'
    forest = models.fit_classifier_model(train_set,
                                         RandomForestClassifier,
                                         data_columns=data_columns,
                                         class_columns=class_col,
                                         bootstrap=True,
                                         compute_importances=None,
                                         criterion='gini',
                                         max_depth=None,
                                         max_features='auto',
                                         max_leaf_nodes=None, min_density=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         n_estimators=1500, n_jobs=-1,
                                         oob_score=True, random_state=None,
                                         verbose=0)
    svm1 = models.fit_classifier_model(train_set,
                                       SVC, data_columns=data_columns,
                                       class_columns=class_col,
                                       class_weight='auto',
                                       kernel='rbf', degree=10, C=1000)
    svm3 = models.fit_classifier_model(train_set, LinearSVC,
                                       data_columns=data_columns,
                                       class_columns=class_col, C=1000)
    knn = models.fit_classifier_model(train_set,
                                      KNeighborsClassifier,
                                      data_columns=data_columns,
                                      class_columns=class_col)
    return grouped_classifier.GroupedClassifier([forest, svm1, svm3, knn])

def train_clusterers(data_set):
    data_columns = data_set.columns[:-2]
    kmeans  = models.fit_clustering_model(data_set, KMeans, data_columns=data_columns)
    dbscan = models.fit_clustering_model(data_set, DBSCAN, data_columns=data_columns)
    return grouped_clusterer.GroupedClusterer([kmeans, dbscan])
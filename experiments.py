import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing._weights import balance_weights
from matplotlib.backends.backend_pdf import PdfPages

import base.data_handler as dh
import base.grouped_classifier as grouped_classifier
import base.grouped_clusterer as grouped_clusterer
import base.models as models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


def __plot_results_inner(results, labels):
    matrices = results['matrix']
    score = results['score']
    crossvalidation = results['crossvalidation']
    figs = grouped_classifier.plot_matrices(matrices)
    figs.append(grouped_classifier.plot_scores(score, labels))
    figs.append(grouped_classifier.plot_crossvalidation(crossvalidation, labels))
    return figs


def plot_results(results, labels=None, save_file=None):
    figs = __plot_results_inner(results, labels)

    if save_file is not None:
        with PdfPages(save_file) as pp:
            for fig in figs:
                pp.savefig(fig)
    plt.show()
    plt.close()


def run_pca_tests(spectra_df, n_components=5):
    '''Runs tests using PCA'''
    kinds = ['ICA', 'PCA', 'Kernel']
    results = {}

    for kind in kinds:
        decomposed = dh.decompose(spectra_df, data_cols=spectra_df.columns[0:-2],
                                  n_components=n_components, kind=kind)
        transformed_train_set, transformed_test_set = dh.split_train_set(decomposed)

        classifiers = train_classifiers(transformed_train_set)
        test_set_data = transformed_test_set.as_matrix(decomposed.columns[0:-2])
        test_set_labels = transformed_test_set['class'].as_matrix().ravel()
        weights = balance_weights(test_set_labels)

        score = classifiers.score_all(test_set_data, test_set_labels)
        result = {'classifiers': classifiers, 'score': score,
                  'matrix': classifiers.confusion_matrix(test_set_data, test_set_labels),
                  'crossvalidation': run_crossvalidation_on_group(classifiers, decomposed)}
        results[kind] = result
    return results


def run_without_pca(spectra_df):
    results = {}
    train_set, test_set = dh.split_train_set(spectra_df)
    classifiers = train_classifiers(train_set)
    test_set_data = test_set.as_matrix(test_set.columns[0:-2])
    test_set_labels = test_set['class'].as_matrix().ravel()
    weights = balance_weights(test_set_labels)
    # print(weights)
    results['classifiers'] = classifiers
    results['score'] = classifiers.score_all(test_set_data, test_set_labels)
    results['matrix'] = classifiers.confusion_matrix(test_set_data, test_set_labels)
    results['crossvalidation'] = run_crossvalidation_on_group(classifiers,
                                                              spectra_df)
    return results


def run_crossvalidation_on_group(group, validation_set):
    data_columns = validation_set.columns[:-2]
    class_col = 'class'
    scores = group.crossvalidation_score(validation_set.as_matrix(data_columns),
                                         validation_set[class_col].as_matrix().ravel())
    return scores


def train_classifiers(train_set, weights=None):
    data_columns = train_set.columns[:-2]
    class_col = 'class'
    forest = models.find_best_params(train_set,
                                     RandomForestClassifier,
                                     data_columns=data_columns,
                                     class_columns=class_col,
                                     parameter_dict={'min_samples_leaf': list(range(1, 5)),
                                                     'min_samples_split': list(range(2, 6)),
                                                     },
                                     bootstrap=True,
                                     n_estimators=2*len(data_columns),
                                     criterion='entropy',
                                     max_depth=None,
                                     max_features='auto',
                                     max_leaf_nodes=None, min_density=None,
                                     n_jobs=-1,
                                     oob_score=True, random_state=None,
                                     verbose=0)
    svm1 = models.find_best_params(train_set,
                                   SVC,
                                   {'C': [2 ** exp for exp in list(range(-5, 10, 2))],
                                    'gamma': [2 ** exp for exp in list(range(-5, 2))] + [0]},
                                   data_columns=data_columns,
                                   class_columns=class_col,
                                   class_weight='auto',
                                   kernel='rbf')

    svm3 = models.find_best_params(train_set, LinearSVC,
                                   parameter_dict={
                                       'C': [2 ** exp for exp in range(-10, 10, 2)]
                                   },
                                   data_columns=data_columns,
                                   class_columns=class_col,
                                   dual=False if len(data_columns) > len(train_set) else True)
    knn = models.find_best_params(train_set,
                                  KNeighborsClassifier,
                                  parameter_dict={ 'n_neighbors': list(range(1, 100, 10))},
                                  data_columns=data_columns,
                                  class_columns=class_col)
    return grouped_classifier.GroupedClassifier([forest, svm1, svm3, knn], labels=["forest", "SVC", "LinearSVC", "KNN"])


def get_feature_importances_using_rdf(train_set):
    data_columns = train_set.columns[:-2]
    class_col = 'class'
    forest = models.fit_classifier_model(train_set,
                                         RandomForestClassifier,
                                         data_columns=data_columns,
                                         class_columns=class_col,
                                         bootstrap=True,
                                         criterion='entropy',
                                         max_depth=None,
                                         max_features='auto',
                                         max_leaf_nodes=None, min_density=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         n_estimators=len(data_columns) * 3, n_jobs=-1,
                                         oob_score=True, random_state=None,
                                         verbose=0)
    return forest.feature_importances_


def train_clusterers(data_set):
    data_columns = data_set.columns[:-2]
    kmeans  = models.fit_clustering_model(data_set, KMeans, data_columns=data_columns)
    dbscan = models.fit_clustering_model(data_set, DBSCAN, data_columns=data_columns)
    return grouped_clusterer.GroupedClusterer([kmeans, dbscan])

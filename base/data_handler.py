import os
import io
import random
import itertools
import pyfits
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA, KernelPCA, FastICA
import sklearn.preprocessing as preprocessing
import sklearn.feature_selection as feature_selection
import matplotlib.pyplot as plt


def load_spectra_from_fits(uri):
    return _to_array(_parse_all_fits(uri))


def process_set(spectra, normalize=True, binning=True, remove_duplicates=True, delimiter=','):
    if binning:
        spectra = _binning(spectra)
    if normalize:
        _normalize(spectra)
    # csv_name = 'processed.csv'
    #_write_fits_csv(spectra, csv_name)
    return spectra


def load_set(uri, format='csv', header=False, delimiter=','):
    return pd.read_csv(uri, header=0 if header else None,
                       sep=None, dtype=None, na_values='?',
                       skipinitialspace=True)


def to_dataframe(spectra_list, class_dict=None):
    indices = [spectrum['id'] for spectrum in spectra_list]
    columns = spectra_list[0]['header']
    # columns.append('label')
    data = [spectrum['data'] for spectrum in spectra_list]
    spectra_df = pd.DataFrame(data=data, columns=columns, index=indices)
    if class_dict is not None:
        classes = [class_dict[index] for index in indices]
        spectra_df.insert(len(spectra_df.columns), 'class', classes)
    return spectra_df


def _to_array(fits_list):
    for fits in fits_list:
        data = []
        header = []
        for length, intensity in fits['data']:
            data.append(intensity)
            header.append(length)
        fits['data'] = data
        fits['header'] = header
    return fits_list


def __spectra_rebinning(fits_list):
    '''Bin the incoming data (expecting two columns [wavelength, intensity]) based on the difference
    between two subsequent points. If the difference between avg of current bin and current point
    exceeds 0.25 we will start new bin

    Note that fits['header'] is array of wavelengths and fits['data'] is array of intensities which
    conforms to the wavelengths.
    '''
    result = []
    # pprint(fits_list)
    #pprint(fits_list[0])
    #pprint(fits_list[0]['data'][0])
    firsts = [x['header'][0] for x in fits_list]
    lasts = [x['header'][-1] for x in fits_list]
    start = min(firsts)
    stop = max(lasts)
    binned_header = np.linspace(start, stop, len(fits_list[0]['header']))
    #print((first_min + first_max) / 2, (last_min + last_max) / 2)
    for fits in fits_list:
        fits_data = fits['data']
        fits_header = fits['header']
        binned_data = np.interp(binned_header, xp=fits_header, fp=fits_data)
        binned_dictionary = {'data': binned_data, 'id': fits['id'], 'header': binned_header}
        result.append(binned_dictionary)
    return result


def _normalize(fits_list, norm='l2'):
    '''normalize data'''
    # min_max_scaler = preprocessing.MinMaxScaler()
    data_list = [fits['data'] for fits in fits_list]
    preprocessed_data = preprocessing.normalize(data_list, norm=norm)
    for idx, item in enumerate(preprocessed_data):
        fits_list[idx]['data'] = item


def _parse_all_fits(uri):
    parsed_fits = []
    classes = None
    current_class = None
    #features = 1997
    for root, dirs, files in os.walk(uri):
        fits_files = [file for file in files if file.endswith('.fits')]
        if len(fits_files) == 0: continue
        for fits_file in fits_files:
            try:
                fits = {}
                fits["data"] = _parse_fits(os.path.join(root, fits_file))
                fits["id"] = os.path.splitext(fits_file)[0]
                parsed_fits.append(fits)
            except:
                print(str(e) + "for :" + str(fits_file))
    # pprint.pprint(parsed_fits)
    return parsed_fits


def _write_csv(data, uri, header=None, separator=',', dtypes=None):
    with io.open(uri, 'w', encoding='utf-8') as out:
        if header is not None or False:
            print("writing header")
            for record in header:
                try:
                    out.write(str(record))
                except TypeError:
                    out.write(unicode(str(record)))
                if (record != header[-1]):
                    out.write(separator)
            out.write('\n')

        for row in data:
            rec_num = 0
            for record in row:
                val = record
                if (dtypes is not None and 'int' in str(dtypes[rec_num])):
                    val = int(val)
                elif (dtypes is not None and 'float' in str(dtypes[rec_num])):
                    val = float(val)
                out.write(str(val))
                if (rec_num != len(row) - 1):
                    out.write(separator)
                rec_num += 1
            out.write('\n')


def _parse_fits(uri):
    fits = pyfits.open(uri, memmap=False)
    dat = fits[1].data
    fits.close()
    return dat.tolist()


def split_train_set(data_frame, ratio=0.2):
    classes = data_frame['class'].as_matrix()
    indices = StratifiedShuffleSplit(classes, test_size=ratio, n_iter=1)
    train_set = None
    test_set = None
    for train, test in indices:
        train_set = data_frame.iloc[train]
        test_set = data_frame.iloc[test]
    return train_set, test_set


def plot_data(data, subdir=''):
    fig = plt.figure()
    out_dir = os.path.normpath('./plots/' + subdir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'
    for column in data:
        data[column].plot()
        plt.savefig(out_dir + column + '.pdf')
        plt.clf()
    plt.close()


def decompose(data, data_cols=None, kind='ICA', n_components=None, iterations=300):
    decompositor = None
    if kind == 'ICA':
        decompositor = FastICA(n_components=n_components, max_iter=iterations)
    elif kind == 'PCA':
        decompositor = PCA(n_components=n_components)
    elif kind == 'Kernel':
        decompositor = KernelPCA(n_components=n_components, max_iter=iterations)
    transformed_data = decompositor.fit_transform(data.as_matrix(data_cols))
    # columns = ['pca{0:0>3}'.format(idx) for idx, value in enumerate(transformed_data, start=0)]
    dataframe = pd.DataFrame(transformed_data, index=data.index)
    dataframe.insert(len(dataframe.columns), 'class', data['class'])
    return dataframe


def select_features(data_frame, percentile):
    """Will select the n_features most important features"""
    data_matrix = data_frame.as_matrix(data_frame.columns[0:-2])
    fs = feature_selection.SelectPercentile(percentile=percentile, score_func=feature_selection.f_classif)
    selected = fs.fit_transform(data_matrix, data_frame['class'].as_matrix())
    new_df = pd.DataFrame(selected, data_frame.index, [data_frame.columns[i] for i in fs.get_support(True)])
    new_df.insert(len(new_df.columns), 'class', data_frame['class'])
    return new_df, list(itertools.compress(fs.scores_, fs.get_support()))


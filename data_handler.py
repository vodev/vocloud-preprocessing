import pyfits
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold, KFold
import sklearn.preprocessing as preprocessing
import os
import io
from pprint import pprint

def process_set(uris, normalize=True, binning=True, delimiter=','):
    all_fits = _parse_all_fits(uris)
    all_fits = _to_array(all_fits)
    if(binning):
        all_fits = _binning(all_fits)
    if normalize:
        _normalize(all_fits)
    #csv_name = 'processed.csv'
    #_write_fits_csv(all_fits, csv_name)
    return all_fits

def load_set(uri, format='csv', header=False, delimiter=','):
    return pd.read_csv(uri, header=0 if header else None, sep=None, dtype=None, na_values='?', skipinitialspace=True)

def to_dataframe(spectra_list):
    index = [spectrum['id'] for spectrum in spectra_list]
    columns = spectra_list[0]['header']
    data = [spectrum['data'] for spectrum in spectra_list]
    df = pd.DataFrame(data=data, columns=columns, index=index)
    return df

def _to_array(fits_list):
    for fits in fits_list:
        data = fits['data']
        arr = []
        header = []
        for length, intensity in data:
            arr.append(intensity)
            header.append(length)
        fits['data'] = arr
        fits['header'] = header
    return fits_list

def _binning(fits_list):
    '''do data binning'''
    result = []
    #pprint(fits_list)
    #pprint(fits_list[0])
    #pprint(fits_list[0]['data'][0])

    first_min = fits_list[0]['header'][0]
    first_max = fits_list[0]['header'][0]
    last_min = fits_list[0]['header'][-1]
    last_max = fits_list[0]['header'][-1]

    for fits in fits_list:
        first_min = min(fits['header'][0], first_min)
        first_max = max(fits['header'][0], first_max)
        last_min = min(fits['header'][-1], last_min)
        last_max = max(fits['header'][-1], last_max)
    first_avg = first_max
    last_avg = last_min
    diff = 0.25
    #print((first_min + first_max) / 2, (last_min + last_max) / 2)
    for fits in fits_list:
        fits_data = fits['data']
        fits_header = fits['header']
        binned_data = []
        binned_header = []
        current_val = first_avg
        it = 0
        columns = 0
        while current_val <= last_avg:
            while fits_header[it] > current_val or fits_header[it + 1] < current_val:
                it += 1
            diff_x = fits_header[it + 1] - fits_header[it]
            diff_y = fits_data[it] - fits_data[it + 1]
            diff_x_val = current_val - fits_header[it]
            div = diff_x_val / diff_x
            binned_header.append(current_val)
            binned_data.append(fits_data[it] - diff_y * div)
            current_val += diff
            columns += 1
        binned_dictionary = {}
        binned_dictionary['data'] = binned_data
        binned_dictionary['id'] = fits['id']
        binned_dictionary['class'] = fits['class']
        binned_dictionary['header'] = binned_header
        result.append(binned_dictionary)
    return result


def _normalize(fits_list):
    '''normalize data'''
    min_max_scaler = preprocessing.MinMaxScaler()
    data_list = [fits['data'] for fits in fits_list]
    preprocessed_data = preprocessing.normalize(data_list, norm='l2')
    for idx, item in enumerate(preprocessed_data):
        fits_list[idx]['data'] = item


def _write_fits_csv(fits_list, name):
    csv_file = io.open(name, mode='w', encoding="utf-8")
    csv_file.write('id,')
    for record in fits_list[0]['data']:
        csv_file.write(str(record[0]))
        csv_file.write(',')
    csv_file.write('class\n')
    for fits in fits_list:
        #print(fits)
        csv_file.write(fits['id'])
        csv_file.write(',')
        for record in fits['data']:
            #print(str(record[1]))
            csv_file.write(str(record[1]))
            csv_file.write(',')
        csv_file.write(fits['class'])
        csv_file.write('\n')
    csv_file.close()


def _parse_all_fits(uri):
    parsed_fits = []
    classes = None
    current_class = None
    features = 1997
    for root, dirs, files in os.walk(uri):
        base = os.path.basename(root)
        #print(base)
        if(root == uri):
            classes = dirs
        elif(base in classes):
            current_class = base
        for fi in files:
            if(fi.endswith('.fits')):
                fits_data = _parse_fits(os.path.join(root, fi))
                if(len(fits_data) != features):
                    continue
                fits = {}
                fits['data'] = fits_data
                fits['id'] = fi[0:-5]
                fits['class'] = current_class
                #pprint.pprint(fits[-1])

                parsed_fits.append(fits)

    #pprint.pprint(parsed_fits)
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
                if(record != header[-1]):
                    out.write(separator)
            out.write('\n')

        for row in data:
            rec_num = 0
            for record in row:
                val = record
                if(dtypes is not None and 'int' in str(dtypes[rec_num])):
                    val = int(val)
                elif(dtypes is not None and 'float' in str(dtypes[rec_num])):
                    val = float(val)
                out.write(str(val))
                if(rec_num != len(row) - 1):
                    out.write(separator)
                rec_num += 1
            out.write('\n')


def _parse_fits(uri):
    fits = pyfits.open(uri, memmap=False)
    dat = fits[1].data
    fits.close()
    return dat.tolist()
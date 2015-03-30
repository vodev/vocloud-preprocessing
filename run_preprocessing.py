import sys
import os
import json
import base.data_handler as dh
def main():
    if len(sys.argv) < 2:
        raise ValueError('Must pass at least one input file')
    for file in sys.argv[1:]:
        run_preprocessing(file)


def run_preprocessing(input_file):
    json_dict = None
    spectra_df = None
    normalize = do_binning = remove_duplicates = False
    with open(input_file, 'r') as f:
        json_dict = json.load(f)
    if 'data_uri' in json_dict:
        spectra_list = dh.load_spectra_from_fits(json_dict['data_uri'])
    else:
        raise ValueError("'data_uri' key is missing. Please specify the data location")
    if 'normalize' in json_dict:
        normalize = json_dict['normalize']
    if 'binning' in json_dict:
        do_binning = json_dict['binning']
    if 'remove_duplicates' in json_dict:
        remove_duplicates = json_dict['remove_duplicates']
    processed_df = dh.to_dataframe(dh.process_set(spectra_list,
                                  normalize=normalize,
                                  binning=do_binning,
                                  remove_duplicates=remove_duplicates))
    if 'select_features' in json_dict:
        select_features_dict = json_dict['select_features']
        processed_df, scores = dh.select_features(processed_df,
                                                  select_features_dict['percentile'])

    if 'decompose' in json_dict:
        decompose_dict = json_dict['decompose']
        processed_df = dh.decompose(data=processed_df,
                                    n_components=decompose_dict['components'],
                                    iterations=decompose_dict['iterations'] if
                                    'iterations' in decompose_dict else 300,
                                    kind=decompose_dict['kind'])
    try:
        os.mkdir("./out")
    except:
        pass
    processed_df.to_csv("./out/" + json_dict['out_file'], header=True, index=True, index_label='id')

if __name__ == '__main__':
    main()

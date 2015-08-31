import sys
import json
import string
import base.data_handler as dh
import os
__LINK_TEMPLATE = string.Template('<option selected id="${spectrum_name}_link">${spectrum_name}</option>\n')
__script_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    if len(sys.argv) < 2:
        raise ValueError('Must pass at least one input file')
    for file in sys.argv[1:]:
        run_preprocessing(file)


def _generate_spectra(spectra):
    with open(__script_dir + "/spectra_list.html.template") as template_file:
        html_template = string.Template(template_file.read())
    spectra_list = []
    for index, spectrum in spectra.iterrows():
        spectrum_link = __LINK_TEMPLATE.substitute({'spectrum_name': str(index),
                                                    'spectrum_name_short': str(index)})
        spectra_list.append(spectrum_link)
    categories = json.dumps(spectra.columns.values.tolist())

    html_code = html_template.substitute(
        {"list": "".join(spectra_list),"cats": categories})
    try:
        spectra.drop("class", axis=1).to_csv("spectra.txt", header=False, index=False, sep=",")
    except ValueError:
        spectra.to_csv("spectra.txt", header=False, index=False, sep=",")
    return html_code

def run_preprocessing(input_file):
    json_dict = None
    normalize = do_binning = remove_duplicates = False
    with open(input_file, 'r') as f:
        json_dict = json.load(f)
    if 'normalize' in json_dict:
        normalize = json_dict['normalize']
    if 'binning' in json_dict:
        do_binning = json_dict['binning']
    if 'remove_duplicates' in json_dict:
        remove_duplicates = json_dict['remove_duplicates']
    classes = {}
    if 'classes_file' in json_dict:
        with open(json_dict['classes_file'], 'r') as ji:
            classes.update(json.load(ji))
    spectra_list = dh.load_spectra_from_fits('.')
    processed_df = dh.to_dataframe(dh.process_set(spectra_list,
                                                  normalize=normalize,
                                                  binning=do_binning,
                                                  remove_duplicates=remove_duplicates),
                                   class_dict=classes)

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
    if 'som_format' in json_dict and json_dict['som_format']:
        processed_df.index.to_series().to_csv("./names.txt", header=False, index=False)
        try:
            processed_df['class'].to_csv("./classes.txt", header=False, index=False)
            processed_df.drop(['class'], axis=1).to_csv("./som.csv", header=False, index=False, sep=" ")
        except KeyError:
            processed_df.to_csv("./som.csv", header=False, index=False, sep=" ")
    processed_df.to_csv("./" + json_dict['out_file'], header=True, index=True, index_label='id')

    html_code = _generate_spectra(processed_df)
    with open("./index.html", "w") as file:
        file.write(html_code)

if __name__ == '__main__':
    main()

# author palicka
# created on 28/01/15
from unittest import TestCase
import numpy as np
import base.data_handler as dh

class TestProcessSet(TestCase):

    def setUp(self):
        self.spectra = []
        number_of_spectra = 100
        wavelength_limits = (6300.0, 6600.0)
        features_limits = (1000, 2000)
        for i in range(number_of_spectra):
            starting_value = wavelength_limits[0] + np.random.random_sample()
            features_length = np.random.randint(features_limits[0], features_limits[1])
            delta = (wavelength_limits[1] - starting_value) / features_length
            header = [wavelength_limits[0] + idx * delta for idx in range(features_length)]
            spectrum = {'id': str(i), 'header': header, 'data': np.random.random_sample(len(header)),
                        'class': np.random.randint(4)}
            self.spectra.append(spectrum)

    def tearDown(self):
        print('tearing down')

    def test_process_set_with_binning(self):
        spectra = dh.process_set(self.spectra, False, True)
        lengths = set()
        for spectrum in spectra:
            lengths.add(len(spectrum['header']))
            self.assertEqual(len(lengths), 1, 'Headers don\'t have equal size')


import random
def generate_spectra(num, weighted_choices, header=None, length=1997):
    spectra_list = []
    population = [val for val, cnt in weighted_choices for i in range(cnt)]
    for i in range(num):
        val = random.choice(population)
        spectra_list.append(generate_spectrum(val))

def generate_spectrum(spectrum_type, length):
    generated_points = []
    if spectrum_type == 0:
        return __spectrum_type_a()
    elif spectrum_type == 1:
        return __spectrum_type_b()
    elif spectrum_type == 2:
        return __spectrum_type_c()
    elif spectrum_type == 3:
        return __spectrum_type_d()
    else:
        return None

def __spectrum_type_a(length, generated_points):
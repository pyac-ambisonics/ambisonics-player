# a class making different HRTF preprocessing algorithms available

class Preprocess:
    
    def __init__(self):
        self.algorithms = ['MagLS', 'TA', 'BiMagLS']
        self.current = 'MagLS'

    # apply the chosen preprocessing algorithm. use MagLS as default
    def apply_preprocessing(self, hrirs, algorithm='MagLS'):
        match algorithm:
            case 'MagLS':
                print("Using MagLS HRTF Preprocessing")
                return self.__mag_ls(hrirs)
            case 'TA':
                print("TA not implemented yet. Using MagLS instead")
                return self.__mag_ls(hrirs)
            case 'BiMagLS':
                print("BiMagLS not implemented yet. Using MagLS instead")
                return self.__mag_ls(hrirs)
            case _:
                print("Using default MagLS algorithm")
                return self.__mag_ls(hrirs)


    def __mag_ls(self, hrirs, cutoff=3000):
        # create frequency axis
        # find cutoff frequency

        # for each direction do the following steps???

            # get fft of original HRTF

            # for each ear:

                # below cutoff: keep original

                # above cutoff: magintude from original, phase from min-phase

                # create time domain by ifft-ing

                # apply time domain to magls_hrir for the current direction

        return hrirs
import numpy as np
import pyfar as pf
import spharpy
from math import utils

# a class making different HRTF preprocessing algorithms available

class HRTF_process:
    
    def __init__(self):
        # the different algorithms available
        self.algorithms = ['LS', 'MagLS', 'TA', 'BiMagLS']
        # the current algorithm
        self.current = 'LS'
        # the number of frequency bins we use for doing FFT
        self.n_bins = 2048

    # apply the chosen preprocessing algorithm. use MagLS as default
    def apply_preprocessing(self, hrirs, sh, algorithm='LS'):
        match algorithm:
            case 'LS':
                print("Using standard spherical harmonics processing (Least Squares)")
                return self.__ls(hrirs, sh)
            case 'MagLS':
                print("Using MagLS HRTF Preprocessing")
                return self.__mag_ls(hrirs, sh)
            case 'TA':
                print("TA not implemented yet. Using MagLS instead")
                return self.__mag_ls(hrirs, sh)
            case 'BiMagLS':
                print("BiMagLS not implemented yet. Using MagLS instead")
                return self.__mag_ls(hrirs, sh)
            case _:
                print("Using default spherical harmonics processing (Least Squares)")
                return self.__ls(hrirs, sh)

    # solves the Least Squares Problem. This means just applying the spherical Harmonics to the HRTF
    def __ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics):
        print("Using LS method: Simple matrix multiplication.")
        return (sh.basis_inv @ hrirs).T

    # apply the magnitude least squares algorithm for better results for low order ambisonics
    # ramp = 0 -> no ramp. ramp = 1 -> default ramp (cutoff * sqrt(2)). ramp > 1 -> specific ramp in freq
    def __mag_ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics, cutoff=3000, ramp=1):
        print("Using MagLS method.")
        
        hrirs_copy = hrirs.copy()
        # make our data frequency data
        hrirs_freq = hrirs.freq_raw
        # find corresponding frequency bin
        freq_idx = hrirs.find_nearest_frequency(cutoff)

        # create lambda value for each frequency: possibly with a ramp up? or just 0/1?
        # with just 0/1 we might have to smooth the phase later
        alpha = np.zeros_like(hrirs_freq, dtype=float)
        alpha[freq_idx:] = 1

        
        # make a ramp up if needed. Use Hanning for smoothness
        if ramp > 0:
            # use default ramp of sqrt(2)
            if ramp == 1:
                ramp = cutoff * np.sqrt(2)
            # find start and stop indices. make sure they are not below 0 or above fs
            half = ramp / 2
            start = hrirs.find_nearest_frequency(cutoff - half)
            # probably unnecessary, since find_nearest_freq should always return valid indices
            # start = max(start, 0) 
            stop = hrirs.find_nearest_frequency(cutoff + half)
            # probably unnecessary, since find_nearest_freq should always return valid indices
            # stop = min(hrirs_freq.size, stop)
            # length
            length = max(stop - start, 0)
            # create a Hanning window, use the half that goes from 0 -> 1
            win = np.hanning(2*length)[length:]
            alpha[start:start+length] = win


        # below cutoff: do simple least squares
        # do regular  least squares -> simple matrix mult
        hrirs_sh = (sh.basis_inv @ hrirs).T
        nm_magls = hrirs_sh.copy()
        Y = sh.basis.copy()
        # for each frequency, starting at the cutoff bin
        for f in range(start, hrirs_freq.size):
            # above cutoff: do least squres magnitude from original, phase from min-phase
            # solve the magnitude only optimization
            for ear in range(2):
                # for each ear, all directional data, and the current frequency: find magls
                nm_magls[ear, :, f] = (
                    alpha[f] * self.__mag_ls_iter(
                        A=Y, 
                        b=hrirs_freq[ear, :, f], 
                        x_prev=nm_magls[ear, :, f-1]
                    )
                    + (1 - alpha[f]) * hrirs_sh[ear, :, f]
                )

                # align phase between bins

        # create time domain by ifft-ing

        # apply time domain to magls_hrir for the current direction

        return hrirs_copy
    
    # the core of the algorithm, this produces the actual iterative least squares evaluation
    def __mag_ls_iter(A, b, x_prev):
        pass
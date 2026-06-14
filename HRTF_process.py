import numpy as np
import pyfar as pf
import spharpy
from scipy.optimize import minimize
import shroom.utils.math_utils as sh_util

# a class making different HRTF preprocessing algorithms available

class HRTF_process:
    
    def __init__(self):
        # the different algorithms available
        self.algorithms = ['LS', 'MagLS', 'TA', 'BiMagLS']
        # the current algorithm
        self.current_algorithm = 'LS'
        # the number of frequency bins we use for doing FFT
        self.n_bins = 2048
        # for each algorithm, a different pre-gain might be necessary.
        self.__gain = {
            'LS': 1., 
            'MagLS': 1.28, 
            'TA': 1., 
            'BiMagLS': 1.28
        }

    def get_gain(self):
        return self.__gain[self.current_algorithm]

    # apply the chosen preprocessing algorithm. use MagLS as default
    def apply_preprocessing(self, hrirs, sh, algorithm='LS'):
        match algorithm:
            case 'LS':
                print("Using standard spherical harmonics processing (Least Squares)")
                self.current_algorithm = algorithm
                return self.__ls(hrirs, sh)
            case 'MagLS':
                print("Using MagLS HRTF Preprocessing")
                self.current_algorithm = algorithm
                return self.__mag_ls(hrirs, sh)
            case 'TA':
                print("TA not implemented yet. Using MagLS instead")
                self.current_algorithm = algorithm
                return self.__mag_ls(hrirs, sh)
            case 'BiMagLS':
                print("BiMagLS not implemented yet. Using MagLS instead")
                self.current_algorithm = algorithm
                return self.__mag_ls(hrirs, sh)
            case _:
                print("Using default spherical harmonics processing (Least Squares)")
                self.current_algorithm = algorithm
                return self.__ls(hrirs, sh)

    # solves the Least Squares Problem. This means just applying the spherical Harmonics to the HRTF
    def __ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics):
        print("Using LS method: Simple matrix multiplication.")
        return (sh.basis_inv @ hrirs).T

    # apply the magnitude least squares algorithm for better results for low order ambisonics
    # ramp = 0 -> no ramp. ramp = 1 -> default ramp (cutoff * (1/sqrt(2))). ramp > 1 -> specific ramp in freq
    def __mag_ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics, cutoff=3000, ramp=1):
        print("Using MagLS method.")
        # make our data frequency data
        hrirs_freq = hrirs.freq_raw.copy()
        # find corresponding frequency bin
        freq_axis = hrirs.frequencies
        stop = hrirs.find_nearest_frequency(cutoff)

        # create alpha value for each frequency: possibly with a ramp up? or just 0/1?
        # with just 0/1 we might have to smooth the phase later
        alpha = np.zeros_like(freq_axis, dtype=float)
        alpha[stop:] = 1
        start = stop

        
        # make a ramp up if needed. Use Hanning for smoothness
        if ramp > 0:
            # use default ramp of f * (1/sqrt(2))
            if ramp == 1:
                ramp = cutoff * (1 / np.sqrt(2))
            # find start and stop indices. find_nearest_freq should always return valid indices
            start = hrirs.find_nearest_frequency(cutoff - ramp)
            stop = hrirs.find_nearest_frequency(cutoff)
            length = max(stop - start, 0)
            # create a Hanning window, use the half that goes from 0 -> 1
            win = np.hanning(2*length)[:length]
            alpha[start:start+length] = win

        # below cutoff: do simple least squares
        # do regular  least squares -> simple matrix mult
        hrirs_sh = (sh.basis_inv @ hrirs).T.freq_raw
        nm_magls = hrirs_sh.copy()
        
        sh_basis = sh.basis.copy()
        # for each ear, for each frequency, starting at the cutoff bin
        print("Solving Magnitude Least Squares for each frequency bin.")
        for ear in range(2):
            for f in range(start, freq_axis.size):
                # for each ear, all directional data, and the current frequency: find magls
                # solve the magnitude only optimization
                # above cutoff: do least squares, magnitude from original, phase from min-phase
                nm_magls[ear, :, f] = (
                    # use shroom library for speed
                    # only returns real numbers????
                    alpha[f] * sh_util.magls(A=sh_basis, 
                                             b=hrirs_freq[:, ear, f], 
                                             x_prev=nm_magls[ear, :, f-1]
                                             )
                    # our own solution
                    # alpha[f] * self.__mag_ls_solver(Y=sh_basis, 
                    #                                 target=hrirs_freq[:, ear, f], 
                    #                                 x_prev=nm_magls[ear, :, f-1]
                    #                                 )
                    + (1 - alpha[f]) * hrirs_sh[ear, :, f]
                )
        
        print("Creating pyfar signal from calculated Magnitude Least Squares frequencies.")
        # create time domain by creating a pyfar Signal from frequency data
        hrirs_sh_time = pf.Signal(nm_magls, hrirs.sampling_rate, n_samples=hrirs.n_samples, domain='freq')
        return hrirs_sh_time
    
    # the core of the algorithm, this produces the actual iterative least squares evaluation
    # but its suuuuuuuuuper slow. 8 minutes for 140 frequency bins
    def __mag_ls_solver(self, Y, target, x_prev):

        x_prev = np.asarray(x_prev, dtype=complex)
        n = x_prev.size

        # stack real and imaginary parts into a signle real vector
        real_array = np.concatenate([x_prev.real, x_prev.imag])

        # loss determines the squared difference between predicted magnitude and target magnitude
        def loss(r):
            # recreate our complex array
            x = r[:n] + 1j * r[n:]
            predicted_mag = np.abs(Y @ x)
            target_mag = np.abs(target)
            return np.sum((predicted_mag - target_mag)**2)
        
        # with minimize we find the least loss in precision. this is relatively slow though
        result = minimize(loss, real_array, method='L-BFGS-B')
        result_x = result.x
        # recreate our complex values
        x_min = result_x[:n] + 1j * result_x[n:]

        # to do: implement phase alignment

        return x_min
